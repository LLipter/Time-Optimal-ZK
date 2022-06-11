use std::iter::repeat_with;
use std::collections::HashMap;
use std::time::Instant;
use std::sync::RwLock;
use itertools::iproduct;
use rand::Rng;
use ff::Field;
use ff::PrimeField;
use ndarray::Array;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::ArrayViewMut;
use ndarray::Dim;
use ndarray::parallel::prelude::*;
use num_traits::Num;
use sprs::MulAcc;
use digest::Digest;
use digest::Output;
use rayon::prelude::*;
use sprs::CsMat;
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

pub fn linear_combination_2_1<F>(
    msg_len: usize, 
    code_len: usize,
    m_2d: &Array<F, Dim<[usize; 2]>>,
    r: &Vec<F>
) -> Array<F, Dim<[usize; 1]>>
where
    F: PrimeField + Num,
{
    assert_eq!(m_2d.shape(), &[code_len, msg_len]);
    assert_eq!(r.len(), msg_len);

    let mut result = Array::<F, _>::zeros((msg_len));
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i1, mut x)| {
            let data = x.first_mut().unwrap();
            for i2 in 0..msg_len {
                *data = data.add(r[i2].mul(m_2d[[i1, i2]]));
            }
            *x.first_mut().unwrap() = *data;
        });
    return result;
}

pub fn linear_combination_2_1_simple_zk<F>(
    msg_len: usize, 
    code_len: usize,
    m_2d: &Array<F, Dim<[usize; 2]>>,
    r: &Vec<F>
) -> Array<F, Dim<[usize; 1]>>
where
    F: PrimeField + Num,
{
    let code_len2 = 2 * code_len;
    assert_eq!(m_2d.shape(), &[code_len2, msg_len]);
    assert_eq!(r.len(), msg_len);

    let mut result = Array::<F, _>::zeros((code_len2));
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i1, mut x)| {
            let data = x.first_mut().unwrap();
            for i2 in 0..msg_len {
                *data = data.add(r[i2].mul(m_2d[[i1, i2]]));
            }
            *x.first_mut().unwrap() = *data;
        });
    return result;
}

pub fn linear_combination_3_2<F>(
    msg_len: usize, 
    code_len: usize,
    m_3d: &Array<F, Dim<[usize; 3]>>,
    r: &Vec<F>
) -> Array<F, Dim<[usize; 2]>>
where
    F: PrimeField + Num,
{
    assert_eq!(m_3d.shape(), &[code_len, code_len, msg_len]);
    assert_eq!(r.len(), msg_len);

    let mut result = Array::<F, _>::zeros((code_len, msg_len));
    result
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
                        *data = data.add(r[i3].mul(m_3d[[i1, i2, i3]]));
                    }
                    *x.first_mut().unwrap() = *data;
                });
        });
    return result;
}

pub fn linear_combination_4_3<F>(
    msg_len: usize, 
    code_len: usize,
    m_4d: &Array<F, Dim<[usize; 4]>>,
    r: &Vec<F>
) -> Array<F, Dim<[usize; 3]>>
where
    F: PrimeField + Num,
{
    assert_eq!(m_4d.shape(), &[code_len, code_len, code_len, msg_len]);
    assert_eq!(r.len(), msg_len);

    let mut result = Array::<F, _>::zeros((code_len, code_len, msg_len));
    result
        .axis_iter_mut(Axis(2))
        .into_par_iter()
        .enumerate()
        .for_each(|(i3, mut xxx)| {
            xxx
                .axis_iter_mut(Axis(1))
                .enumerate()
                .for_each(|(i2, mut xx)| {
                    xx
                        .axis_iter_mut(Axis(0))
                        .enumerate()
                        .for_each(|(i1, mut x)| {
                            let data = x.first_mut().unwrap();
                            for i4 in 0..msg_len {
                                *data = data.add(r[i4].mul(m_4d[[i1, i2, i3, i4]]));
                            }
                            *x.first_mut().unwrap() = *data;
                        });
                });
        });
    return result;
}

pub fn merkle_tree_commit_2d<F, D>(
    msg_len: usize, 
    code_len: usize,
    m_2d: &Array<F, Dim<[usize; 2]>>
) -> Vec<Output<D>>
where
    F: PrimeField,
    D: Digest,
{
    assert_eq!(m_2d.shape(), &[code_len, msg_len]);

    let mut hashes_vec = Vec::<Output<D>>::new();
    let item_no = code_len;
    let np2 = next_pow_2(item_no);
    hashes_vec.resize_with(2*np2-1, Default::default);
    (&mut hashes_vec)
        .into_par_iter()
        .enumerate()
        .filter(|(i, _)| i >= &(np2-1) && i < &(np2-1+item_no))
        .for_each(|(i, x)| {
        let mut digest = D::new();
        for i2 in 0..msg_len {
            let i1 = i-(np2-1);
            digest.update(m_2d[[i1, i2]].to_repr());
        }
        *x = digest.finalize();
    });
    build_merkle_tree::<D>(&mut hashes_vec, np2);
    return hashes_vec;
}

pub fn merkle_tree_commit_3d<F, D>(
    msg_len: usize, 
    code_len: usize,
    m_3d: &Array<F, Dim<[usize; 3]>>
) -> Vec<Output<D>>
where
    F: PrimeField,
    D: Digest,
{
    assert_eq!(m_3d.shape(), &[code_len, code_len, msg_len]);

    let mut hashes_vec = Vec::<Output<D>>::new();
    let item_no = code_len * code_len;
    let np2 = next_pow_2(item_no);
    hashes_vec.resize_with(2*np2-1, Default::default);
    (&mut hashes_vec)
        .into_par_iter()
        .enumerate()
        .filter(|(i, _)| i >= &(np2-1) && i < &(np2-1+item_no))
        .for_each(|(i, x)| {
        let mut digest = D::new();
        for i3 in 0..msg_len {
            let i1 = (i-(np2-1)) % code_len;
            let i2 = (i-(np2-1)) / code_len;
            digest.update(m_3d[[i1, i2, i3]].to_repr());
        }
        *x = digest.finalize();
    });
    build_merkle_tree::<D>(&mut hashes_vec, np2);
    return hashes_vec;
}

pub fn merkle_tree_commit_4d<F, D>(
    msg_len: usize, 
    code_len: usize,
    m_4d: &Array<F, Dim<[usize; 4]>>
) -> Vec<Output<D>>
where
    F: PrimeField,
    D: Digest,
{
    assert_eq!(m_4d.shape(), &[code_len, code_len, code_len, msg_len]);

    let mut hashes_vec = Vec::<Output<D>>::new();
    let item_no = code_len * code_len * code_len;
    let np2 = next_pow_2(item_no);
    hashes_vec.resize_with(2*np2-1, Default::default);
    (&mut hashes_vec)
        .into_par_iter()
        .enumerate()
        .filter(|(i, _)| i >= &(np2-1) && i < &(np2-1+item_no))
        .for_each(|(i, x)| {
        let mut digest = D::new();
        for i4 in 0..msg_len {
            let i1 = (i-(np2-1)) % code_len;
            let i2 = (i-(np2-1)) / code_len % code_len;
            let i3 = (i-(np2-1)) / code_len / code_len;
            digest.update(m_4d[[i1, i2, i3, i4]].to_repr());
        }
        *x = digest.finalize();
    });
    build_merkle_tree::<D>(&mut hashes_vec, np2);
    return hashes_vec;
}

pub fn check_linear_combination_2_1<F>(
    msg_len: usize, 
    code_len: usize,
    m_2d: &Array<F, Dim<[usize; 2]>>,
    m_1d: &Array<F, Dim<[usize; 1]>>,
    r: &Vec<F>,
    precodes: &Vec<CsMat<F>>,
    postcodes: &Vec<CsMat<F>>,
    i1: usize,
) -> bool
where
    F: PrimeField + Num + MulAcc,
{
    assert_eq!(m_2d.shape(), &[code_len, msg_len]);
    assert_eq!(m_1d.shape(), &[msg_len]);
    assert_eq!(r.len(), msg_len);

    let mut msg = Vec::<F>::with_capacity(code_len);
    for i1 in 0..msg_len {
        msg.push(m_1d[[i1]]);
    }
    msg.resize(code_len, <F as Field>::zero());
    encode(&mut msg, precodes, postcodes);
    let mut s = <F as Field>::zero();
    for i2 in 0..msg_len {
        s = s.add(r[i2].mul(m_2d[[i1, i2]]));
    }
    return s == msg[i1];
}

pub fn check_linear_combination_2_1_simple_zk<F>(
    msg_len: usize, 
    code_len: usize,
    m_2d: &Array<F, Dim<[usize; 2]>>,
    m_1d: &Array<F, Dim<[usize; 1]>>,
    r: &Vec<F>,
    precodes: &Vec<CsMat<F>>,
    postcodes: &Vec<CsMat<F>>,
    i1: usize,
) -> bool
where
    F: PrimeField + Num + MulAcc,
{
    let code_len2 = 2 * code_len;
    assert_eq!(m_2d.shape(), &[code_len2, msg_len]);
    assert_eq!(m_1d.shape(), &[code_len2]);
    assert_eq!(r.len(), msg_len);

    let mut msg = Vec::<F>::with_capacity(code_len);
    for i1 in 0..msg_len {
        msg.push(m_1d[[i1]].sub(m_1d[[i1+code_len]]));
    }
    msg.resize(code_len, <F as Field>::zero());
    encode(&mut msg, precodes, postcodes);
    for i1 in 0..code_len {
        msg[i1] = msg[i1].add(m_1d[[i1+code_len]]);
        msg.push(m_1d[[i1+code_len]]);
    }
    let mut s = <F as Field>::zero();
    for i2 in 0..msg_len {
        s = s.add(r[i2].mul(m_2d[[i1, i2]]));
    }
    return s == msg[i1];
}

pub fn check_linear_combination_3_2<F>(
    msg_len: usize, 
    code_len: usize,
    m_3d: &Array<F, Dim<[usize; 3]>>,
    m_2d: &Array<F, Dim<[usize; 2]>>,
    r: &Vec<F>,
    precodes: &Vec<CsMat<F>>,
    postcodes: &Vec<CsMat<F>>,
    i1: usize,
    i2: usize,
) -> bool
where
    F: PrimeField + Num + MulAcc,
{
    assert_eq!(m_3d.shape(), &[code_len, code_len, msg_len]);
    assert_eq!(m_2d.shape(), &[code_len, msg_len]);
    assert_eq!(r.len(), msg_len);

    let mut msg = Vec::<F>::with_capacity(code_len);
    for i2 in 0..msg_len {
        msg.push(m_2d[[i1, i2]]);
    }
    msg.resize(code_len, <F as Field>::zero());
    encode(&mut msg, &precodes, &postcodes);
    let mut s = <F as Field>::zero();
    for i3 in 0..msg_len {
        s = s.add(r[i3].mul(m_3d[[i1, i2, i3]]));
    }
    return s == msg[i2];
}

pub fn check_linear_combination_4_3<F>(
    msg_len: usize, 
    code_len: usize,
    m_4d: &Array<F, Dim<[usize; 4]>>,
    m_3d: &Array<F, Dim<[usize; 3]>>,
    r: &Vec<F>,
    precodes: &Vec<CsMat<F>>,
    postcodes: &Vec<CsMat<F>>,
    i1: usize,
    i2: usize,
    i3: usize,
) -> bool
where
    F: PrimeField + Num + MulAcc,
{
    assert_eq!(m_4d.shape(), &[code_len, code_len, code_len, msg_len]);
    assert_eq!(m_3d.shape(), &[code_len, code_len, msg_len]);
    assert_eq!(r.len(), msg_len);

    let mut msg = Vec::<F>::with_capacity(code_len);
    for i3 in 0..msg_len {
        msg.push(m_3d[[i1, i2, i3]]);
    }
    msg.resize(code_len, <F as Field>::zero());
    encode(&mut msg, &precodes, &postcodes);
    let mut s = <F as Field>::zero();
    for i4 in 0..msg_len {
        s = s.add(r[i4].mul(m_4d[[i1, i2, i3, i4]]));
    }
    return s == msg[i3];
}

pub fn commit_2_dim<F, C, D>(
    coef_no: usize, 
    msg_len: usize, 
    code_len: usize, 
    seed: u64,
    test_no: usize,
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
    let m1 = linear_combination_2_1::<F>(msg_len, code_len, &m0, &r1);

    // commit to m0
    let hashes_m0 = merkle_tree_commit_2d::<F, D>(msg_len, code_len, &m0);

    let committed_time = Instant::now();

    // verifier has access to r1, m1, m0.root
    let mut m0_map = HashMap::<usize, Output<D>>::new();
    m0_map.insert(0, hashes_m0[0].clone());
    // sample idx
    let mut idx_1 = Vec::<usize>::new();
    idx_1.resize_with(test_no, || rng.gen_range(0..code_len));
    (0..test_no).into_par_iter().for_each(|i| {
        let i1 = idx_1[i];

        assert!(
            check_linear_combination_2_1::<F>(
                msg_len, code_len, 
                &m0, &m1, &r1, 
                &precodes, &postcodes, 
                i1
            )
        );

    });

    let m0_map_rwlock = RwLock::new(m0_map); 
    (0..test_no).into_par_iter().for_each(|i| {
        let i1 = idx_1[i];

        // verify the merkle path for m0
        let mut digest = D::new();
        for i2 in 0..msg_len {
            digest.update(m0[[i1, i2]].to_repr());
        }
        let cur_hash = digest.finalize();
        let item_no = code_len;
        let np2 = next_pow_2(item_no);
        let idx = i1 + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, idx, &m0_map_rwlock, &hashes_m0));
        // println!("test {} passed", i+1);
    });

    let verified_time = Instant::now();

    println!("t:2 coef_no:{:?} msg_len:{:?} code_len:{:?} test_no:{:?}", coef_no, msg_len, code_len, test_no);
    println!("commit_time: {} ms", committed_time.duration_since(start_time).as_millis());
    println!("verify_time: {} ms", verified_time.duration_since(committed_time).as_millis());
    println!("total_time: {} ms\n", verified_time.duration_since(start_time).as_millis());
}

pub fn commit_2_dim_simple_zk<F, C, D>(
    coef_no: usize, 
    msg_len: usize, 
    code_len: usize, 
    seed: u64,
    test_no: usize,
)
where
    F: PrimeField + Num + MulAcc,
    C: CodeSpecification,
    D: Digest,
{
    let mut rng = rand::thread_rng();
    let code_len2 = 2 * code_len;

    // generate codes
    let (precodes, postcodes) = generate::<F, C>(msg_len, seed);

    // M0: 2N * m
    let mut m0 = Array::<F, _>::zeros((code_len2, msg_len));
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
            let mut rng = rand::thread_rng();
            let mut msg = x.to_vec();
            msg.resize(code_len, <F as Field>::zero());
            encode(&mut msg, &precodes, &postcodes);
            for i in 0..code_len {
                let e = F::random(&mut rng);
                msg[i] = msg[i].add(e);
                msg.push(e);
            }
            for i1 in 0..code_len2 {
                x[i1] = msg[i1];
            }
        });

    // M1: 2N
    let m1 = linear_combination_2_1_simple_zk::<F>(msg_len, code_len, &m0, &r1);

    // commit to m0
    let hashes_m0 = merkle_tree_commit_2d::<F, D>(msg_len, code_len2, &m0);

    let committed_time = Instant::now();

    // verifier has access to r1, m1, m0.root
    let mut m0_map = HashMap::<usize, Output<D>>::new();
    m0_map.insert(0, hashes_m0[0].clone());
    // sample idx
    let mut idx_1 = Vec::<usize>::new();
    idx_1.resize_with(test_no, || rng.gen_range(0..code_len2));
    (0..test_no).into_par_iter().for_each(|i| {
        let i1 = idx_1[i];

        assert!(
            check_linear_combination_2_1_simple_zk::<F>(
                msg_len, code_len, 
                &m0, &m1, &r1, 
                &precodes, &postcodes, 
                i1
            )
        );

    });

    let m0_map_rwlock = RwLock::new(m0_map); 
    (0..test_no).into_par_iter().for_each(|i| {
        let i1 = idx_1[i];

        // verify the merkle path for m0
        let mut digest = D::new();
        for i2 in 0..msg_len {
            digest.update(m0[[i1, i2]].to_repr());
        }
        let cur_hash = digest.finalize();
        let item_no = code_len2;
        let np2 = next_pow_2(item_no);
        let idx = i1 + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, idx, &m0_map_rwlock, &hashes_m0));
        // println!("test {} passed", i+1);
    });

    let verified_time = Instant::now();

    println!("simple zk t:2 coef_no:{:?} msg_len:{:?} code_len:{:?} test_no:{:?}", coef_no, msg_len, code_len, test_no);
    println!("commit_time: {} ms", committed_time.duration_since(start_time).as_millis());
    println!("verify_time: {} ms", verified_time.duration_since(committed_time).as_millis());
    println!("total_time: {} ms\n", verified_time.duration_since(start_time).as_millis());
}

pub fn commit_3_dim<F, C, D>(
    coef_no: usize, 
    msg_len: usize, 
    code_len: usize, 
    seed: u64,
    test_no: usize,
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
    let m1 = linear_combination_3_2::<F>(msg_len, code_len, &m0, &r1);
    // M2: m
    let m2 = linear_combination_2_1::<F>(msg_len, code_len, &m1, &r2);


    // commit to m0
    let hashes_m0 = merkle_tree_commit_3d::<F, D>(msg_len, code_len, &m0);
    // commit to m1
    let hashes_m1 = merkle_tree_commit_2d::<F, D>(msg_len, code_len, &m1);

    let committed_time = Instant::now();

    // verifier has access to r1, r2, m2, m0.root, m1.root
    let mut m0_map = HashMap::<usize, Output<D>>::new();
    m0_map.insert(0, hashes_m0[0].clone());
    let mut m1_map = HashMap::<usize, Output<D>>::new();
    m1_map.insert(0, hashes_m1[0].clone());
    // sample idx
    let mut idx_1 = Vec::<usize>::new();
    idx_1.resize_with(test_no, || rng.gen_range(0..code_len));
    let mut idx_2 = Vec::<usize>::new();
    idx_2.resize_with(test_no, || rng.gen_range(0..code_len));
    (0..test_no).into_par_iter().for_each(|i| {
        let i1 = idx_1[i];
        let i2 = idx_2[i];
        assert!(
            check_linear_combination_3_2::<F>(
                msg_len, code_len, 
                &m0, &m1, &r1, 
                &precodes, &postcodes, 
                i1, i2
            )
        );
        assert!(
            check_linear_combination_2_1::<F>(
                msg_len, code_len, 
                &m1, &m2, &r2, 
                &precodes, &postcodes, 
                i1
            )
        );
    });
    
    let m0_map_rwlock = RwLock::new(m0_map);
    let m1_map_rwlock = RwLock::new(m1_map);
    (0..test_no).into_par_iter().for_each(|i| {
        let i1 = idx_1[i];
        let i2 = idx_2[i];

        // verify the merkle path for m0
        let mut digest = D::new();
        for i3 in 0..msg_len {
            digest.update(m0[[i1, i2, i3]].to_repr());
        }
        let cur_hash = digest.finalize();
        let item_no = code_len * code_len;
        let np2 = next_pow_2(item_no);
        let idx = i1 + i2 * code_len + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, idx, &m0_map_rwlock, &hashes_m0));
    
        // verify the merkle path for m1
        let mut digest = D::new();
        for i2 in 0..msg_len {
            digest.update(m1[[i1, i2]].to_repr());
        }
        let cur_hash = digest.finalize();
        let item_no = code_len;
        let np2 = next_pow_2(item_no);
        let idx = i1 + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, idx, &m1_map_rwlock, &hashes_m1));
        // println!("test {} passed", i+1);
    });

    let verified_time = Instant::now();

    println!("t:3 coef_no:{:?} msg_len:{:?} code_len:{:?} test_no:{:?}", coef_no, msg_len, code_len, test_no);
    println!("commit_time: {} ms", committed_time.duration_since(start_time).as_millis());
    println!("verify_time: {} ms", verified_time.duration_since(committed_time).as_millis());
    println!("total_time: {} ms\n", verified_time.duration_since(start_time).as_millis());
}

pub fn commit_4_dim<F, C, D>(
    coef_no: usize, 
    msg_len: usize, 
    code_len: usize, 
    seed: u64,
    test_no: usize,
)
where
    F: PrimeField + Num + MulAcc,
    C: CodeSpecification,
    D: Digest,
{
    let mut rng = rand::thread_rng();

    // generate codes
    let (precodes, postcodes) = generate::<F, C>(msg_len, seed);

    // M0: N * N * N * m
    let mut m0 = Array::<F, _>::zeros((code_len, code_len, code_len, msg_len));
    // generate random coefficient: m * m * m * m
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
    let mut r3 = Vec::<F>::new();
    r3.resize_with(msg_len, || F::random(&mut rng));

    // encode for axis 0
    m0
        .axis_iter_mut(Axis(3))
        .into_par_iter()
        .enumerate()
        .for_each(|(i4, mut xxx)| {
            xxx
                .axis_iter_mut(Axis(2))
                .enumerate()
                .filter(|(i, _)| i < &(msg_len))
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
        });
    // encode for axis 1
    m0
        .axis_iter_mut(Axis(3))
        .into_par_iter()
        .enumerate()
        .for_each(|(i4, mut xxx)| {
            xxx
                .axis_iter_mut(Axis(2))
                .enumerate()
                .filter(|(i, _)| i < &(msg_len))
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
        });
    // encode for axis 2
    m0
        .axis_iter_mut(Axis(3))
        .into_par_iter()
        .enumerate()
        .for_each(|(i4, mut xxx)| {
            xxx
                .axis_iter_mut(Axis(1))
                .enumerate()
                .for_each(|(i2, mut xx)| {
                    xx
                        .axis_iter_mut(Axis(0))
                        .enumerate()
                        .for_each(|(i1, mut x)| {
                            let mut msg = x.to_vec();
                            msg.resize(code_len, <F as Field>::zero());
                            encode(&mut msg, &precodes, &postcodes);
                            for i3 in 0..code_len {
                                x[i3] = msg[i3];
                            }
                        });
                });
        });

    // M1: N * N * m
    let m1 = linear_combination_4_3::<F>(msg_len, code_len, &m0, &r1);
    // M2: N * m
    let m2 = linear_combination_3_2::<F>(msg_len, code_len, &m1, &r2);
    // M3: m
    let m3 = linear_combination_2_1::<F>(msg_len, code_len, &m2, &r3);


    // commit to m0
    let hashes_m0 = merkle_tree_commit_4d::<F, D>(msg_len, code_len, &m0);
    // commit to m1
    let hashes_m1 = merkle_tree_commit_3d::<F, D>(msg_len, code_len, &m1);
    // commit to m2
    let hashes_m2 = merkle_tree_commit_2d::<F, D>(msg_len, code_len, &m2);

    let committed_time = Instant::now();

    // verifier has access to r1, r2, r3, m3, m0.root, m1.root, m2.root
    let mut m0_map = HashMap::<usize, Output<D>>::new();
    m0_map.insert(0, hashes_m0[0].clone());
    let mut m1_map = HashMap::<usize, Output<D>>::new();
    m1_map.insert(0, hashes_m1[0].clone());
    let mut m2_map = HashMap::<usize, Output<D>>::new();
    m2_map.insert(0, hashes_m2[0].clone());
    // sample idx
    let mut idx_1 = Vec::<usize>::new();
    idx_1.resize_with(test_no, || rng.gen_range(0..code_len));
    let mut idx_2 = Vec::<usize>::new();
    idx_2.resize_with(test_no, || rng.gen_range(0..code_len));
    let mut idx_3 = Vec::<usize>::new();
    idx_3.resize_with(test_no, || rng.gen_range(0..code_len));
    (0..test_no).into_par_iter().for_each(|i| {
        let i1 = idx_1[i];
        let i2 = idx_2[i];
        let i3 = idx_3[i];

        assert!(
            check_linear_combination_4_3::<F>(
                msg_len, code_len, 
                &m0, &m1, &r1, 
                &precodes, &postcodes, 
                i1, i2, i3
            )
        );
        assert!(
            check_linear_combination_3_2::<F>(
                msg_len, code_len, 
                &m1, &m2, &r2, 
                &precodes, &postcodes, 
                i1, i2
            )
        );
        assert!(
            check_linear_combination_2_1::<F>(
                msg_len, code_len, 
                &m2, &m3, &r3, 
                &precodes, &postcodes, 
                i1
            )
        );
    });
    
    let m0_map_rwlock = RwLock::new(m0_map);
    let m1_map_rwlock = RwLock::new(m1_map);
    let m2_map_rwlock = RwLock::new(m2_map);
    (0..test_no).into_par_iter().for_each(|i| {
        let i1 = idx_1[i];
        let i2 = idx_2[i];
        let i3 = idx_3[i];

        // verify the merkle path for m0
        let mut digest = D::new();
        for i4 in 0..msg_len {
            digest.update(m0[[i1, i2, i3, i4]].to_repr());
        }
        let cur_hash = digest.finalize();
        let item_no = code_len * code_len * code_len;
        let np2 = next_pow_2(item_no);
        let idx = i1 + i2 * code_len + i3 * code_len * code_len + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, idx, &m0_map_rwlock, &hashes_m0));

        // verify the merkle path for m1
        let mut digest = D::new();
        for i3 in 0..msg_len {
            digest.update(m1[[i1, i2, i3]].to_repr());
        }
        let cur_hash = digest.finalize();
        let item_no = code_len * code_len;
        let np2 = next_pow_2(item_no);
        let idx = i1 + i2 * code_len + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, idx, &m1_map_rwlock, &hashes_m1));
    
        // verify the merkle path for m2
        let mut digest = D::new();
        for i2 in 0..msg_len {
            digest.update(m2[[i1, i2]].to_repr());
        }
        let cur_hash = digest.finalize();
        let item_no = code_len;
        let np2 = next_pow_2(item_no);
        let idx = i1 + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, idx, &m2_map_rwlock, &hashes_m2));
        // println!("test {} passed", i+1);
    });

    let verified_time = Instant::now();

    println!("t:4 coef_no:{:?} msg_len:{:?} code_len:{:?} test_no:{:?}", coef_no, msg_len, code_len, test_no);
    println!("commit_time: {} ms", committed_time.duration_since(start_time).as_millis());
    println!("verify_time: {} ms", verified_time.duration_since(committed_time).as_millis());
    println!("total_time: {} ms\n", verified_time.duration_since(start_time).as_millis());
}