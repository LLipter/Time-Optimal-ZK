use std::iter::repeat_with;
use std::collections::HashMap;
use rand::Rng;
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

// pub fn get_1d_index(dim: &Vec<usize>, idx: &Vec<usize>) -> usize {
//     assert_eq!(dim.len(), idx.len());
//     return dim.iter().zip(idx).map(|(a, b)| a * b).sum();
// }

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
    assert_eq!(msg_len * msg_len, coef_no);
    let np2 = next_pow_2(code_len);

    // generate random coefficient: m * m
    let mut coef = Array::<F, _>::zeros((msg_len, msg_len));
    let mut rng = rand::thread_rng();
    for a in 0..msg_len {
        for b in 0..msg_len {
            coef[[a, b]] = F::random(&mut rng);
        }
    }

    // random linear combination
    let mut r1 = Vec::<F>::new();
    r1.resize_with(msg_len, || F::random(&mut rng));

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

    // m1
    let mut m1 = Array::<F, _>::zeros((code_len));
    for a in 0..code_len {
        for x in 0..msg_len {
            m1[[a]] = m1[[a]].add(r1[x].mul(m0[[a, x]]));
        }
    }

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


    // verifier has access to r1, m1, m0.root
    let mut m0_map = HashMap::<usize, Output<D>>::new();
    m0_map.insert(0, hashes_m0[0].clone());
    for i in 0..test_no {
        // sample idx
        let i1 = rng.gen_range(0..code_len);
        let mut msg = Vec::<F>::with_capacity(code_len);
        for x in 0..msg_len {
            msg.push(m1[[x]]);
        }
        // println!("{:?}", msg);
        msg.resize(code_len, <F as Field>::zero());
        encode(&mut msg, &precodes, &postcodes);
        let mut s1 = <F as Field>::zero();
        for k in 0..msg_len {
            s1 = s1.add(r1[k].mul(m0[[i1, k]]));
        }
        assert_eq!(s1, msg[i1]);

        // verify the merkle path for m0
        let mut digest = D::new();
        for k in 0..msg_len {
            digest.update(m0[[i1, k]].to_repr());
        }
        let mut cur_hash = digest.finalize();
        let mut idx = i1 + np2 - 1;
        while idx > 0 {
            match m0_map.get(&idx) {
                None => {
                    m0_map.insert(idx, cur_hash.clone());
                },
                Some(h) => assert!(cur_hash.eq(h)),
            }

            let mut digest = D::new();
            if idx % 2 == 0 {
                digest.update(&hashes_m0[idx-1]);
                digest.update(&cur_hash);
            }else{
                digest.update(&cur_hash);
                digest.update(&hashes_m0[idx+1]);
            }
            cur_hash = digest.finalize();
            idx = (idx - 1) / 2;
        }
        assert!(cur_hash.eq(&m0_map[&0]));
        println!("test {} passed", i+1);
    }
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
    assert_eq!(msg_len * msg_len * msg_len, coef_no);
    let np2 = next_pow_2(code_len);

    // generate random coefficient: m * m * m
    let mut coef = Array::<F, _>::zeros((msg_len, msg_len, msg_len));
    let mut rng = rand::thread_rng();
    for a in 0..msg_len {
        for b in 0..msg_len {
            for c in 0..msg_len {
                coef[[a, b, c]] = F::random(&mut rng);
            }
        }
    }


    // random linear combination
    let mut r1 = Vec::<F>::new();
    r1.resize_with(msg_len, || F::random(&mut rng));
    let mut r2 = Vec::<F>::new();
    r2.resize_with(msg_len, || F::random(&mut rng));

    // generate codes
    let (precodes, postcodes) = generate::<F, C>(msg_len, seed);
    
    // M0: N * N * m
    let mut m0 = Array::<F, _>::zeros((code_len, code_len, msg_len));
    
    // encode for axis 0
    for a in 0..msg_len {
        for b in 0..msg_len {
            let mut msg = Vec::<F>::with_capacity(code_len);
            for x in 0..msg_len {
                msg.push(coef[[x, a, b]]);
            }
            msg.resize(code_len, <F as Field>::zero());
            encode(&mut msg, &precodes, &postcodes);
            for x in 0..code_len {
                m0[[x, a, b]] = msg[x];
            }
        }
    }
    // encode for axis 1
    for a in 0..code_len {
        for b in 0..msg_len {
            let mut msg = Vec::<F>::with_capacity(code_len);
            for x in 0..msg_len {
                msg.push(m0[[a, x, b]]);
            }
            msg.resize(code_len, <F as Field>::zero());
            encode(&mut msg, &precodes, &postcodes);
            for x in msg_len..code_len {
                m0[[a, x, b]] = msg[x];
            }
        }
    }
    // println!("{:?}", m0);

    // m1
    let mut m1 = Array::<F, _>::zeros((code_len, code_len));
    for a in 0..code_len {
        for b in 0..code_len {
            for x in 0..msg_len {
                m1[[a, b]] = m1[[a, b]].add(r1[x].mul(m0[[a, b, x]]));
            }
        }
    }

    // m2
    let mut m2 = Array::<F, _>::zeros((code_len));
    for a in 0..code_len {
        for x in 0..msg_len {
            m2[[a]] = m2[[a]].add(r2[x].mul(m1[[a, x]]));
        }
    }

    // // commit to m0
    // let mut hashes_m0 = Vec::<Output<D>>::new();
    // hashes_m0.resize_with(np2-1, Default::default);
    // for i in 0..code_len {
    //     let mut digest = D::new();
    //     for j in 0..msg_len {
    //         digest.update(m0[[i, j]].to_repr());
    //     }
    //     hashes_m0.push(digest.finalize());
    // }
    // // build merkle tree
    // hashes_m0.resize_with(2*np2-1, Default::default);
    // build_merkle_tree::<D>(&mut hashes_m0, np2);


    // // verifier has access to r1, m1, m0.root
    // let mut m0_map = HashMap::<usize, Output<D>>::new();
    // m0_map.insert(0, hashes_m0[0].clone());
    // for i in 0..test_no {
    //     // sample idx
    //     let i1 = rng.gen_range(0..code_len);
    //     let mut msg = Vec::<F>::with_capacity(code_len);
    //     for x in 0..msg_len {
    //         msg.push(m1[[x]]);
    //     }
    //     // println!("{:?}", msg);
    //     msg.resize(code_len, <F as Field>::zero());
    //     encode(&mut msg, &precodes, &postcodes);
    //     let mut s1 = <F as Field>::zero();
    //     for k in 0..msg_len {
    //         s1 = s1.add(r1[k].mul(m0[[i1, k]]));
    //     }
    //     assert_eq!(s1, msg[i1]);

    //     // verify the merkle path for m0
    //     let mut digest = D::new();
    //     for k in 0..msg_len {
    //         digest.update(m0[[i1, k]].to_repr());
    //     }
    //     let mut cur_hash = digest.finalize();
    //     let mut idx = i1 + np2 - 1;
    //     while idx > 0 {
    //         match m0_map.get(&idx) {
    //             None => {
    //                 m0_map.insert(idx, cur_hash.clone());
    //             },
    //             Some(h) => assert!(cur_hash.eq(h)),
    //         }

    //         let mut digest = D::new();
    //         if idx % 2 == 0 {
    //             digest.update(&hashes_m0[idx-1]);
    //             digest.update(&cur_hash);
    //         }else{
    //             digest.update(&cur_hash);
    //             digest.update(&hashes_m0[idx+1]);
    //         }
    //         cur_hash = digest.finalize();
    //         idx = (idx - 1) / 2;
    //     }
    //     assert!(cur_hash.eq(&m0_map[&0]));
    //     println!("test {} passed", i+1);
    // }
}