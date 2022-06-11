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
use crate::helper::next_pow_2;
use crate::helper::linear_combination_2_1;
use crate::helper::linear_combination_3_2;
use crate::helper::linear_combination_4_3;
use crate::merkle::build_merkle_tree;
use crate::merkle::check_merkle_path;
use crate::merkle::merkle_tree_commit_2d;
use crate::merkle::merkle_tree_commit_3d;
use crate::merkle::merkle_tree_commit_4d;
use crate::codespec::CodeSpecification;
use crate::codegen::generate;
use crate::encode::encode;

pub fn check_linear_combination_2_1_simple_zk<F>(
    msg_len: usize, 
    code_len: usize,
    m_2d: &Array<F, Dim<[usize; 2]>>,
    m_1d: &Array<F, Dim<[usize; 1]>>,
    m_1d_pad: &Array<F, Dim<[usize; 1]>>,
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
    assert_eq!(m_1d_pad.shape(), &[code_len]);
    assert_eq!(r.len(), msg_len);

    let mut msg = Vec::<F>::with_capacity(code_len);
    for i1 in 0..msg_len {
        msg.push(m_1d[[i1]].sub(m_1d_pad[[i1]]));
    }
    msg.resize(code_len, <F as Field>::zero());
    encode(&mut msg, precodes, postcodes);
    let mut s = <F as Field>::zero();
    for i2 in 0..msg_len {
        s = s.add(r[i2].mul(m_2d[[i1, i2]]));
    }
    return s == msg[i1].add(m_1d_pad[[i1]]);
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

    // m0: N * m
    let mut m0 = Array::<F, _>::zeros((code_len, msg_len));
    // generate random coefficient: m * m
    m0.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });
    
    let start_time = Instant::now();
    
    // m0_pad: N * m
    let mut m0_pad = Array::<F, _>::zeros((code_len, msg_len));
    // generate random pad: N * m
    m0_pad.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });

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
                x[i1] = msg[i1].add(m0_pad[[i1, i2]]);
            }
        });

    // m1: m
    let m1 = linear_combination_2_1::<F>(msg_len, code_len, &m0, &r1, msg_len);
    // m1_pad: N
    let m1_pad = linear_combination_2_1::<F>(msg_len, code_len, &m0_pad, &r1, code_len);

    // commit to m0, m0_pad
    let hashes_m0 = merkle_tree_commit_2d::<F, D>(msg_len, code_len, &m0);
    let hashes_m0_pad = merkle_tree_commit_2d::<F, D>(msg_len, code_len, &m0_pad);
    
    let committed_time = Instant::now();
    
    // verifier has access to r1, m1, m1_pad, m0.root, m0_pad.root
    let mut m0_map = HashMap::<usize, Output<D>>::new();
    m0_map.insert(0, hashes_m0[0].clone());
    let mut m0_pad_map = HashMap::<usize, Output<D>>::new();
    m0_pad_map.insert(0, hashes_m0_pad[0].clone());

    // sample idx
    let mut idx_1 = Vec::<usize>::new();
    idx_1.resize_with(test_no, || rng.gen_range(0..code_len));
    (0..test_no).into_par_iter().for_each(|i| {
        let i1 = idx_1[i];
        
        assert!(
            check_linear_combination_2_1_simple_zk::<F>(
                msg_len, code_len, 
                &m0, &m1, &m1_pad, &r1, 
                &precodes, &postcodes, 
                i1
            )
        );
        
    });
    
    let m0_map_rwlock = RwLock::new(m0_map);
    let m0_pad_map_rwlock = RwLock::new(m0_pad_map);
    (0..test_no).into_par_iter().for_each(|i| {
        let i1 = idx_1[i];
        let item_no = code_len;
        let np2 = next_pow_2(item_no);
        let idx = i1 + np2 - 1;

        // verify the merkle path for m0
        let mut digest = D::new();
        for i2 in 0..msg_len {
            digest.update(m0[[i1, i2]].to_repr());
        }
        let cur_hash = digest.finalize();
        assert!(check_merkle_path::<D>(cur_hash, idx, &m0_map_rwlock, &hashes_m0));

        // verify the merkle path for m0_pad
        let mut digest = D::new();
        for i2 in 0..msg_len {
            digest.update(m0_pad[[i1, i2]].to_repr());
        }
        let cur_hash = digest.finalize();
        assert!(check_merkle_path::<D>(cur_hash, idx, &m0_pad_map_rwlock, &hashes_m0_pad));
        // println!("test {} passed", i+1);
    });

    let verified_time = Instant::now();

    println!("simple zk t:2 coef_no:{:?} msg_len:{:?} code_len:{:?} test_no:{:?}", coef_no, msg_len, code_len, test_no);
    println!("commit_time: {} ms", committed_time.duration_since(start_time).as_millis());
    println!("verify_time: {} ms", verified_time.duration_since(committed_time).as_millis());
    println!("total_time: {} ms\n", verified_time.duration_since(start_time).as_millis());
}