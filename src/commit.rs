use std::iter::repeat_with;
use std::collections::HashMap;
use std::time::Instant;
use itertools::iproduct;
use rand::Rng;
use ff::Field;
use ff::PrimeField;
use ndarray::Array;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::ArrayViewMut;
use ndarray::parallel::prelude::*;
use num_traits::Num;
use sprs::MulAcc;
use digest::Digest;
use digest::Output;
use rayon::prelude::*;
use crate::codespec::CodeSpecification;
use crate::codegen::generate;
use crate::encode::encode;
use crate::helper::next_pow_2;
use crate::helper::build_merkle_tree;
use crate::helper::check_merkle_path;

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

pub fn commit_2_dim<F, C, D>(
    coef_no: usize, 
    msg_len: usize, 
    code_len: usize, 
    seed: u64,
    test_no: usize
)
where
    F: PrimeField + Num + MulAcc,
    C: CodeSpecification,
    D: Digest,
{
    let mut rng = rand::thread_rng();

    // generate codes
    let (precodes, postcodes) = generate::<F, C>(msg_len, seed);

    // M0: N * m
    let mut m0 = Array::<F, _>::zeros((code_len, msg_len));
    // generate random coefficient: m * m
    m0.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });

    let start_time = Instant::now();

    // random linear combination
    let mut r1 = Vec::<F>::new();
    r1.resize_with(msg_len, || F::random(&mut rng));

    // encode for axis 0
    m0
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(i2, mut x)| {
            let mut msg = x.to_vec();
            msg.resize(code_len, <F as Field>::zero());
            encode(&mut msg, &precodes, &postcodes);
            for i1 in 0..code_len {
                x[i1] = msg[i1];
            }
    });

    // M1: m
    let mut m1 = Array::<F, _>::zeros((msg_len));
    m1
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i1, mut x)| {
            let data = x.first_mut().unwrap();
            for i2 in 0..msg_len {
                *data = data.add(r1[i2].mul(m0[[i1, i2]]));
            }
            *x.first_mut().unwrap() = *data;
    });

    // commit to m0
    let mut hashes_m0 = Vec::<Output<D>>::new();
    let item_no = code_len;
    let np2 = next_pow_2(item_no);
    hashes_m0.resize_with(2*np2-1, Default::default);
    (&mut hashes_m0)
        .into_par_iter()
        .enumerate()
        .filter(|(i, _)| i >= &(np2-1) && i < &(np2-1+item_no))
        .for_each(|(i, x)| {
        let mut digest = D::new();
        for i2 in 0..msg_len {
            let i1 = i-(np2-1);
            digest.update(m0[[i1, i2]].to_repr());
        }
        *x = digest.finalize();
    });
    build_merkle_tree::<D>(&mut hashes_m0, np2);

    let committed_time = Instant::now();

    // verifier has access to r1, m1, m0.root
    let mut m0_map = HashMap::<usize, Output<D>>::new();
    m0_map.insert(0, hashes_m0[0].clone());
    for i in 0..test_no {
        // sample idx
        let i1 = rng.gen_range(0..code_len);

        let mut msg = Vec::<F>::with_capacity(code_len);
        for i1 in 0..msg_len {
            msg.push(m1[[i1]]);
        }

        msg.resize(code_len, <F as Field>::zero());
        encode(&mut msg, &precodes, &postcodes);
        let mut s = <F as Field>::zero();
        for i2 in 0..msg_len {
            s = s.add(r1[i2].mul(m0[[i1, i2]]));
        }
        assert_eq!(s, msg[i1]);

        // verify the merkle path for m0
        let mut digest = D::new();
        for i2 in 0..msg_len {
            digest.update(m0[[i1, i2]].to_repr());
        }
        let cur_hash = digest.finalize();
        let item_no = code_len;
        let np2 = next_pow_2(item_no);
        let idx = i1 + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, idx, &mut m0_map, &hashes_m0));
        // println!("test {} passed", i+1);
    }

    let verified_time = Instant::now();

    println!("commit_time: {} ms", committed_time.duration_since(start_time).as_millis());
    println!("verify_time: {} ms", verified_time.duration_since(committed_time).as_millis());
    println!("total_time: {} ms\n", verified_time.duration_since(start_time).as_millis());
}

pub fn commit_3_dim<F, C, D>(
    coef_no: usize, 
    msg_len: usize, 
    code_len: usize, 
    seed: u64,
    test_no: usize
)
where
    F: PrimeField + Num + MulAcc,
    C: CodeSpecification,
    D: Digest,
{
    let mut rng = rand::thread_rng();

    // generate codes
    let (precodes, postcodes) = generate::<F, C>(msg_len, seed);

    // M0: N * N * m
    let mut m0 = Array::<F, _>::zeros((code_len, code_len, msg_len));
    // generate random coefficient: m * m * m
    m0.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });

    let start_time = Instant::now();

    // random linear combination
    let mut r1 = Vec::<F>::new();
    r1.resize_with(msg_len, || F::random(&mut rng));
    let mut r2 = Vec::<F>::new();
    r2.resize_with(msg_len, || F::random(&mut rng));

    // encode for axis 0
    m0
        .axis_iter_mut(Axis(2))
        .into_par_iter()
        .enumerate()
        .for_each(|(i3, mut xx)| {
            xx
                .axis_iter_mut(Axis(1))
                .enumerate()
                .filter(|(i, _)| i < &(msg_len))
                .for_each(|(i2, mut x)| {
                    let mut msg = x.to_vec();
                    msg.resize(code_len, <F as Field>::zero());
                    encode(&mut msg, &precodes, &postcodes);
                    for i1 in 0..code_len {
                        x[i1] = msg[i1];
                    }
                });
    });
    // encode for axis 1
    m0
        .axis_iter_mut(Axis(2))
        .into_par_iter()
        .enumerate()
        .for_each(|(i3, mut xx)| {
            xx
                .axis_iter_mut(Axis(0))
                .enumerate()
                .for_each(|(i1, mut x)| {
                    let mut msg = x.to_vec();
                    msg.resize(code_len, <F as Field>::zero());
                    encode(&mut msg, &precodes, &postcodes);
                    for i2 in 0..code_len {
                        x[i2] = msg[i2];
                    }
                });
    });

    // M1: N * m
    let mut m1 = Array::<F, _>::zeros((code_len, msg_len));
    m1
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(i2, mut xx)| {
            xx
                .axis_iter_mut(Axis(0))
                .enumerate()
                .for_each(|(i1, mut x)| {
                    let data = x.first_mut().unwrap();
                    for i3 in 0..msg_len {
                        *data = data.add(r1[i3].mul(m0[[i1, i2, i3]]));
                    }
                    *x.first_mut().unwrap() = *data;
                });
    });

    // M2: m
    let mut m2 = Array::<F, _>::zeros((msg_len));
    m2
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i1, mut x)| {
            let data = x.first_mut().unwrap();
            for i2 in 0..msg_len {
                *data = data.add(r2[i2].mul(m1[[i1, i2]]));
            }
            *x.first_mut().unwrap() = *data;
        });


    // commit to m0
    let mut hashes_m0 = Vec::<Output<D>>::new();
    let item_no = code_len * code_len;
    let np2 = next_pow_2(item_no);
    hashes_m0.resize_with(2*np2-1, Default::default);
    (&mut hashes_m0)
        .into_par_iter()
        .enumerate()
        .filter(|(i, _)| i >= &(np2-1) && i < &(np2-1+item_no))
        .for_each(|(i, x)| {
        let mut digest = D::new();
        for i3 in 0..msg_len {
            let i1 = (i-(np2-1)) % code_len;
            let i2 = (i-(np2-1)) / code_len;
            digest.update(m0[[i1, i2, i3]].to_repr());
        }
        *x = digest.finalize();
    });
    build_merkle_tree::<D>(&mut hashes_m0, np2);

    // commit to m1
    let mut hashes_m1 = Vec::<Output<D>>::new();
    let item_no = code_len;
    let np2 = next_pow_2(item_no);
    hashes_m1.resize_with(2*np2-1, Default::default);
    (&mut hashes_m1)
        .into_par_iter()
        .enumerate()
        .filter(|(i, _)| i >= &(np2-1) && i < &(np2-1+item_no))
        .for_each(|(i, x)| {
        let mut digest = D::new();
        for i2 in 0..msg_len {
            let i1 = i-(np2-1);
            digest.update(m1[[i1, i2]].to_repr());
        }
        *x = digest.finalize();
    });
    build_merkle_tree::<D>(&mut hashes_m1, np2);    

    let committed_time = Instant::now();

    // verifier has access to r1, r2, m2, m0.root, m1.root
    let mut m0_map = HashMap::<usize, Output<D>>::new();
    m0_map.insert(0, hashes_m0[0].clone());
    let mut m1_map = HashMap::<usize, Output<D>>::new();
    m1_map.insert(0, hashes_m1[0].clone());
    for i in 0..test_no {
        // sample idx
        let i1 = rng.gen_range(0..code_len);
        let i2 = rng.gen_range(0..code_len);

        let mut msg = Vec::<F>::with_capacity(code_len);
        for i2 in 0..msg_len {
            msg.push(m1[[i1, i2]]);
        }
        msg.resize(code_len, <F as Field>::zero());
        encode(&mut msg, &precodes, &postcodes);
        let mut s = <F as Field>::zero();
        for i3 in 0..msg_len {
            s = s.add(r1[i3].mul(m0[[i1, i2, i3]]));
        }
        assert_eq!(s, msg[i2]);
        msg.clear();
        for i1 in 0..msg_len {
            msg.push(m2[[i1]]);
        }
        msg.resize(code_len, <F as Field>::zero());
        encode(&mut msg, &precodes, &postcodes);
        let mut s = <F as Field>::zero();
        for i2 in 0..msg_len {
            s = s.add(r2[i2].mul(m1[[i1, i2]]));
        }
        assert_eq!(s, msg[i1]);

        // verify the merkle path for m0
        let mut digest = D::new();
        for i3 in 0..msg_len {
            digest.update(m0[[i1, i2, i3]].to_repr());
        }
        let cur_hash = digest.finalize();
        let item_no = code_len * code_len;
        let np2 = next_pow_2(item_no);
        let idx = i1 + i2 * code_len + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, idx, &mut m0_map, &hashes_m0));

        // verify the merkle path for m1
        let mut digest = D::new();
        for i2 in 0..msg_len {
            digest.update(m1[[i1, i2]].to_repr());
        }
        let cur_hash = digest.finalize();
        let item_no = code_len;
        let np2 = next_pow_2(item_no);
        let idx = i1 + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, idx, &mut m1_map, &hashes_m1));
        // println!("test {} passed", i+1);
    }

    let verified_time = Instant::now();

    println!("commit_time: {} ms", committed_time.duration_since(start_time).as_millis());
    println!("verify_time: {} ms", verified_time.duration_since(committed_time).as_millis());
    println!("total_time: {} ms\n", verified_time.duration_since(start_time).as_millis());
}