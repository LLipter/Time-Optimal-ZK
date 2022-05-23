use std::iter::repeat_with;
use ff::Field;
use ff::PrimeField;
use ndarray::Array;
use ndarray::Axis;
use num_traits::Num;
use sprs::MulAcc;
use digest::Digest;
use digest::Output;
use crate::codespec::CodeSpecification;
use crate::codegen::generate;
use crate::encode::encode;
use crate::helper::next_pow_2;
use crate::helper::build_merkle_tree;

/// generate random coeffs of length 2^`log_len`
pub fn random_coeffs<Ft: Field>(log_len: usize) -> Vec<Ft> {
    use std::io::{self, Write};

    let mut rng = rand::thread_rng();
    let mut out = io::stderr();
    let spc = 1 << (if log_len > 6 { log_len - 6 } else { log_len });

    let ret = repeat_with(|| Ft::random(&mut rng))
        .enumerate()
        .take(1 << log_len)
        .inspect(|(c, _)| {
            if c % spc == 0 {
                out.write_all(b".").unwrap();
                out.flush().unwrap();
            }
        })
        .map(|(_, v)| v)
        .collect();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
    ret
}

/// generate random coeffs of length `len`
pub fn random_coeffs_from_length<F>(len: usize) -> Vec<F>
where
    F: PrimeField,
{
    let mut rng = rand::thread_rng();
    return repeat_with(|| F::random(&mut rng)).take(len).collect();
}

pub fn get_1d_index(dim: &Vec<usize>, idx: &Vec<usize>) -> usize {
    assert_eq!(dim.len(), idx.len());
    return dim.iter().zip(idx).map(|(a, b)| a * b).sum();
}

// pub fn commit_t_dim<F, C>(coef_no: usize, t: usize, msg_len: usize, code_len: usize, seed: u64)
// where
//     F: PrimeField,
//     C: CodeSpecification,
// {
//     assert_eq!(msg_len.pow(t as u32), coef_no);

//     let coef = random_coeffs_from_length::<F>(coef_no);
//     let mut M0 = Vec::<F>::new();
//     let M0_len = code_len.pow((t-1) as u32) * msg_len;
//     M0.resize_with(M0_len, F::zero);
//     // println!("{:?}", M0);
//     println!("{}", M0_len);
//     let mut M0_dim = Vec::new();
//     for _ in 0..t-1{
//         M0_dim.push(code_len);
//     }
//     M0_dim.push(msg_len);
//     println!("{:?}", M0_dim);


//     let (precodes, postcodes) = generate::<F, C>(msg_len, seed);
// }

pub fn commit_2_dim<F, C, D>(coef_no: usize, msg_len: usize, code_len: usize, seed: u64)
where
    F: PrimeField + Num + MulAcc,
    C: CodeSpecification,
    D: Digest,
{
    assert_eq!(msg_len * msg_len, coef_no);
    let np2 = next_pow_2(code_len);

    // generate random coefficient: m * m
    let mut coef = Array::<F, _>::zeros((msg_len, msg_len));
    let mut rng = rand::thread_rng();
    for i in 0..msg_len {
        for j in 0..msg_len {
            coef[[i, j]] = F::random(&mut rng);
        }
    }

    // generate codes
    let (precodes, postcodes) = generate::<F, C>(msg_len, seed);
    
    // M0: N * m
    let mut m0 = Array::<F, _>::zeros((code_len, msg_len));
    
    // encode for axis 0
    for a in 0..msg_len {
        let mut msg = Vec::<F>::with_capacity(code_len);
        for x in 0..msg_len {
            msg.push(coef[[x, a]]);
        }
        msg.resize(code_len, <F as Field>::zero());
        encode(&mut msg, &precodes, &postcodes);
        for x in 0..code_len {
            m0[[x, a]] = msg[x];
        }
    }
    // println!("{:?}", m0);


    // commit to m0
    let mut hashes_m0 = Vec::<Output<D>>::new();
    hashes_m0.resize_with(np2-1, Default::default);
    for i in 0..code_len {
        let mut digest = D::new();
        for j in 0..msg_len {
            digest.update(m0[[i, j]].to_repr());
        }
        hashes_m0.push(digest.finalize());
    }
    // build merkle tree
    hashes_m0.resize_with(2*np2-1, Default::default);
    build_merkle_tree::<D>(&mut hashes_m0, np2);

    // for i in 0..hashes_m0.len() {
    //     println!("{}: {:?}", i, hashes_m0[i]);
    // }

    // random linear combination 1
    let mut r1 = Vec::<F>::new();
    r1.resize_with(msg_len, || F::random(&mut rng));
    // println!("{:?}", r1);
    
    // m1
    let mut m1 = Array::<F, _>::zeros((code_len));
    for a in 0..code_len {
        for x in 0..msg_len {
            m1[[a]] = m1[[a]].add(r1[x].mul(m0[[a, x]]));
        }
    }
    // println!("{:?}", m1);

    // commit to m1
    let mut hashes_m1 = Vec::<Output<D>>::new();
    for _ in 0..(np2-1) {
        hashes_m1.push(D::digest(F::random(&mut rng).to_repr()));
    }
    for i in 0..code_len {
        let mut digest = D::new();
        hashes_m1.push(D::digest(m1[[i]].to_repr()));
    }
    for _ in hashes_m1.len()..(2*np2-1) {
        hashes_m1.push(D::digest(F::random(&mut rng).to_repr()));
    }
    for i in (1..(np2-1)).step_by(2).rev() {
        // build merkle tree
        let mut digest = D::new();
        digest.update(&hashes_m1[i]);
        digest.update(&hashes_m1[i+1]);
        hashes_m1[i/2] = digest.finalize();
    }
}