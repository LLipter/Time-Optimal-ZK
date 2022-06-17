use sprs::CsMat;
use sprs::MulAcc;
use ff::Field;
use ff::PrimeField;
use ndarray::ArrayView;
use ndarray::linalg::Dot;
use num_traits::Num;
use rand::thread_rng;
use rand::seq::SliceRandom;
use crate::codegen::generate;
use crate::codegen::generate_rev;
use crate::helper::degree_bound;
use crate::codespec::CodeSpecification;

// given a set of precodes and postcodes, output length of codeword
pub fn codeword_length<F>(precodes: &[CsMat<F>], postcodes: &[CsMat<F>]) -> usize
where
    F: PrimeField,
{
    assert!(!precodes.is_empty());
    assert_eq!(precodes.len(), postcodes.len());

    // input
    precodes[0].cols()
    // R-S result
        + postcodes.last().unwrap().cols()
    // precode outputs (except input to R-S, which is not part of codeword)
        + precodes.iter().take(precodes.len() - 1).map(|pc| pc.rows()).sum::<usize>()
    // postcode outputs
        + postcodes.iter().map(|pc| pc.rows()).sum::<usize>()
}

/// encode a vector given a code of corresponding length
pub fn encode<F, T>(mut xi: T, precodes: &[CsMat<F>], postcodes: &[CsMat<F>])
where
    F: PrimeField + Num + MulAcc,
    T: AsMut<[F]>,
{
    // check sizes
    assert_eq!(xi.as_mut().len(), codeword_length(precodes, postcodes));

    // compute precodes all the way down
    let mut in_start = 0usize;
    for precode in precodes.iter().take(precodes.len() - 1) {
        // compute matrix-vector product
        let in_end = in_start + precode.cols();
        let (in_arr, out_arr) = xi.as_mut().split_at_mut(in_end);
        out_arr[..precode.rows()].copy_from_slice(
            precode
                .dot(&ArrayView::from(&in_arr[in_start..]))
                .as_slice()
                .unwrap(),
        );

        in_start = in_end;
    }

    // base-case code: Reed-Solomon
    let (mut in_start, mut out_start) = {
        // first, evaluate last precode into temporary storage
        let precode = precodes.last().unwrap();
        let in_end = in_start + precode.cols();
        let in_arr = precode
            .dot(&ArrayView::from(&xi.as_mut()[in_start..in_end]))
            .into_raw_vec();

        // now evaluate Reed-Solomon code on the result
        let out_end = in_end + postcodes.last().unwrap().cols();
        reed_solomon(in_arr.as_ref(), &mut xi.as_mut()[in_end..out_end]);

        (in_end + precode.rows(), out_end)
    };

    for (precode, postcode) in precodes.iter().rev().zip(postcodes.iter().rev()) {
        // move input pointer backward
        in_start -= precode.rows();

        // compute matrix-vector product
        let (in_arr, out_arr) = xi.as_mut().split_at_mut(out_start);
        out_arr[..postcode.rows()].copy_from_slice(
            postcode
                .dot(&ArrayView::from(&in_arr[in_start..]))
                .as_slice()
                .unwrap(),
        );

        out_start += postcode.rows();
    }

    assert_eq!(in_start, precodes[0].cols());
    assert_eq!(out_start, xi.as_mut().len());
}

/// reverse-encode a vector given a code of corresponding length
pub fn encode_rev<F, T>(mut data: T, precodes: &[CsMat<F>], postcodes: &[CsMat<F>])
where
    F: PrimeField + Num + MulAcc,
    T: AsMut<[F]>,
{
    // println!("precodes:");
    // for precode in precodes.iter() {
    //     println!("{} {}", precode.cols(), precode.rows());
    // }
    // println!("postcodes:");
    // for postcode in postcodes.iter() {
    //     println!("{} {}", postcode.cols(), postcode.rows());
    // }

    // check sizes
    assert_eq!(data.as_mut().len(), precodes[0].rows() + postcodes[0].rows() + postcodes[0].cols());

    let data_mut = data.as_mut();

    let mut x_start = 0;
    let mut z_start = precodes[0].rows();
    let mut v_start = precodes[0].rows() + postcodes[0].rows();
    let mut xzv_stack = Vec::<(usize, usize, usize)>::new();
    for (idx, postcode) in postcodes.iter().enumerate() {
        if idx != 0 {
            x_start = z_start;
            z_start += precodes[idx].rows();
            v_start -= postcodes[idx].cols();
        }
        // println!("{} {} {}", x_start, z_start, v_start);
        xzv_stack.push((x_start, z_start, v_start));
        let v_len = postcode.cols();
        let z_len = postcode.rows();
        let v_reversed = postcode.dot(&ArrayView::from(&data_mut[v_start..(v_start+v_len)]));
        for i in 0..z_len {
            data_mut[z_start + i] = data_mut[z_start + i].add(v_reversed[i]);
        }

    }

    // base-case code: Reed-Solomon
    let base_len_from = v_start - z_start;
    let base_len_to = postcodes.last().unwrap().cols();
    let vandermonde_start = z_start;
    // println!("{} {} vandermonde_start: {}", base_len_from, base_len_to, vandermonde_start);
    let mut vandermonde_result = Vec::<F>::new();
    vandermonde_result.resize_with(base_len_to, || <F as Field>::zero());
    let mut val_i = <F as Field>::zero();
    for i in 0..base_len_from { // n
        val_i = val_i.add(<F as Field>::one());
        let mut val = <F as Field>::one();
        for j in 0..base_len_to { // k
            vandermonde_result[j] = vandermonde_result[j].add(
                val.mul(data_mut[vandermonde_start+i])
            );
            val = val.mul(val_i);
        }
    }
    for i in 0..base_len_to {
        data_mut[vandermonde_start+i] = vandermonde_result[i];
    }

    for (precode, (x_start, z_start, v_start)) in precodes.iter().rev().zip(xzv_stack.iter().rev()) {
        // println!("{} {} {}", x_start, z_start, v_start);
        let z_len = precode.cols();
        let z_reversed = precode.dot(&ArrayView::from(&data_mut[*z_start..(z_start+z_len)]));
        let x_len = precode.rows();
        for i in 0..x_len {
            data_mut[x_start + i] = data_mut[x_start + i].add(z_reversed[i]);
        }
    }
}

pub fn test_reverse_encoding<F, C>()
where
    F: PrimeField + Num + MulAcc,
    C: CodeSpecification,
{
    let msg_len = 1024;
    let code_len = 1762;
    
    let (precodes, postcodes) = generate::<F, C>(msg_len, 0);
    let mut matrix1 = Vec::<Vec<F>>::new();
    for i in 0..msg_len {
        let mut codeword = Vec::<F>::new();
        codeword.resize_with(code_len, || <F as Field>::zero());
        codeword[i] = <F as Field>::one();
        encode::<F, _>(&mut codeword, &precodes, &postcodes);
        matrix1.push(codeword);
    }

    let (precodes, postcodes) = generate_rev::<F, C>(code_len, 0);
    let mut matrix2 = Vec::<Vec<F>>::new();
    for i in 0..code_len {
        let mut codeword = Vec::<F>::new();
        codeword.resize_with(code_len, || <F as Field>::zero());
        codeword[i] = <F as Field>::one();
        encode_rev::<F, _>(&mut codeword, &precodes, &postcodes);
        matrix2.push(codeword);
    }

    for i in 0..msg_len {
        for j in 0..code_len {
            assert_eq!(matrix1[i][j], matrix2[j][i]);
        }
    }
}

// Compute Reed-Solomon encoding using Vandermonde matrix
fn reed_solomon<F>(xi: &[F], xo: &mut [F])
where
    F: PrimeField,
{
    let mut x = <F as Field>::one();
    for r in xo.as_mut().iter_mut() {
        *r = <F as Field>::zero();
        for j in (0..xi.len()).rev() {
            *r *= x;
            *r += xi[j];
        }
        x += <F as Field>::one();
    }
}

pub fn encode_zk<F, C>(
    mut msg: &mut Vec<F>, 
    precodes: &[CsMat<F>], 
    postcodes: &[CsMat<F>],
)
where
    F: PrimeField + Num + MulAcc,
    C: CodeSpecification,
{
    encode(&mut msg, &precodes, &postcodes);

    let code_len = codeword_length(precodes, postcodes);
    // let degree = degree_bound(1.0/rate, 256, code_len);
    let degree = 3;
    
    // generate random graph
    let mut rng = rand::thread_rng();
    let mut permute: Vec<usize> = (0..code_len).collect();
    let mut graph = Vec::<Vec<usize>>::new();
    graph.resize_with(code_len, || Vec::new());
    for _ in 0..degree {
        permute.shuffle(&mut rng);
        for (i, j) in permute.iter().enumerate() {
            graph[i].push(*j);
        }
    }

    // step1: redistribute
    let mut redistributed = Vec::<F>::new();
    for i in 0..code_len {
        for j in 0..degree {
            redistributed.push(msg[graph[i][j]]);
        }
    }
    let redistributed_len = redistributed.len();

    // step2: randomlize
    let mut randomlized = Vec::<F>::new();
    for i in 0..(redistributed_len / degree) {
        let mut random_block = Vec::<Vec<F>>::new();
        random_block.resize_with(degree, || Vec::<F>::new());
        for i in 0..degree {
            for _ in 0..degree {
                random_block[i].push(F::random(&mut rng));
            }
        }
        for j in 0..degree {
            let mut result = <F as Field>::zero();
            for k in 0..degree {
                result = result.add(
                    redistributed[i * degree + k].mul(
                        random_block[k][j]
                    )
                );
            }
            randomlized.push(result);
        }
    }
    
    // println!("{:?}", randomlized);
    // println!("{:?}", redistributed);



    // TODO: step 3 reverse encoding
}