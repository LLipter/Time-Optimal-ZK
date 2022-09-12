use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;
use digest::Digest;
use digest::Output;
use ff::Field;
use ff::PrimeField;
use ndarray::Array;
use ndarray::Dim;
use ndarray::Axis;
use ndarray::parallel::prelude::*;
use num_traits::Num;
use sprs::MulAcc;
use rand::Rng;
use crate::codespec::CodeSpecification;
use crate::codegen::generate;
use crate::encode::codeword_length;
use crate::encode::encode;
use crate::helper::next_pow_2;
use crate::merkle::build_merkle_tree;
use crate::merkle::check_merkle_path;


pub fn generate_ternary_vector<F>(
    size: usize,
) -> Array<F, Dim<[usize; 1]>>
where
    F: PrimeField + Num + MulAcc,
{

    let mut result = Array::<F, _>::zeros((size));
    result.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        let data: u32 = rng.gen_range(0..=2);
        if data == 0 {
            *x = <F as Field>::zero();
            *x = (*x).sub(<F as Field>::one());
        }else if data == 1{
            *x = <F as Field>::zero();
        }else if data == 2{
            *x = <F as Field>::one();
        }
    });

    return result;
    
}

pub fn fill_H_array<F>(
    H: &mut Vec::<F>,
    n: usize,
    m: usize,
    lambda: usize,
    array1: &Array<F, Dim<[usize; 1]>>,
    array2: &Array<F, Dim<[usize; 1]>>,
    array3: &Array<F, Dim<[usize; 1]>>,
    array4: &Array<F, Dim<[usize; 1]>>,
)
where
    F: PrimeField + Num + MulAcc,
{
    assert_eq!(array1.len(), m);
    assert_eq!(array2.len(), m);
    assert_eq!(array3.len(), n);
    assert_eq!(array4.len(), lambda);
    H
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)|{
            if i < m {
                *x = array1[i];
            }else if i < 2 * m {
                *x = array2[i-m];
            }else if i < 2 * m + n {
                *x = array3[i-2*m];
            }else if i < 2 * m + n + lambda {
                *x = array4[i-2*m-n];
            }
        });
}

pub fn merkle_tree_commit_lwe<F, D>(
    code_len: usize,
    H2: &Vec::<F>,
    H1: &Vec::<F>,
    H0: &Vec::<F>,
) -> Vec<Output<D>>
where
    F: PrimeField,
    D: Digest,
{
    assert_eq!(code_len, H2.len());
    assert_eq!(code_len, H1.len());
    assert_eq!(code_len, H0.len());

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
        let idx = i-(np2-1);
        digest.update(H2[idx].to_repr());
        digest.update(H1[idx].to_repr());
        digest.update(H0[idx].to_repr());
        *x = digest.finalize();
    });
    build_merkle_tree::<D>(&mut hashes_vec, np2);
    return hashes_vec;
}

pub fn ternary_lwe<F, C, D>(
    n: usize,
    m: usize,
    lambda: usize,
    seed: u64,
)
where
    F: PrimeField + Num + MulAcc,
    C: CodeSpecification,
    D: Digest,
{
    let zero = <F as Field>::zero();
    let one = <F as Field>::one();
    let two = one.add(one);
    let three = two.add(one);
    let mut rng = rand::thread_rng();

    // generate codes
    let msg_len: usize = 2 * m + n + lambda;
    let (precodes, postcodes) = generate::<F, C>(msg_len, seed);
    let code_len = codeword_length::<F>(&precodes, &postcodes);

    // A: n * m
    let mut A = Array::<F, _>::zeros((n, m));
    A.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });

    // s: m
    let s = generate_ternary_vector::<F>(m);
    // e: n
    let e = generate_ternary_vector::<F>(n);

    // u: n
    let mut u = A.dot(&s);
    u
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = data.add(e[i]);
            *x.first_mut().unwrap() = *data;
        });

    let start_time = Instant::now();

    // t: m
    let mut t = Array::<F, _>::zeros(m);
    t.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });

    let mut v2 = Array::<F, _>::zeros(m);
    v2
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = t[i].mul(t[i]).mul(t[i]);
            *x.first_mut().unwrap() = *data;
        });

    let mut v1 = Array::<F, _>::zeros(m);
    v1
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = three.mul(s[i]).mul(t[i]).mul(t[i]);
            *x.first_mut().unwrap() = *data;
        });

    let mut v0 = Array::<F, _>::zeros(m);
    v0
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = three.mul(s[i]).mul(s[i]).sub(one).mul(t[i]);
            *x.first_mut().unwrap() = *data;
        });

    

    // At: n
    let At = A.dot(&t);

    let mut w2 = Array::<F, _>::zeros(n);
    w2
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = zero.sub(At[i].mul(At[i]).mul(At[i]));
            *x.first_mut().unwrap() = *data;
        });

    let mut w1 = Array::<F, _>::zeros(n);
    w1
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = three.mul(e[i]).mul(At[i]).mul(At[i]);
            *x.first_mut().unwrap() = *data;
        });

    let mut w0 = Array::<F, _>::zeros(n);
    w0
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = one.sub(
                three.mul(e[i]).mul(e[i])
            ).mul(At[i]);
            *x.first_mut().unwrap() = *data;
        });

    // r0, r1, r2: lambda
    let mut r0 = Array::<F, _>::zeros(lambda);
    let mut r1 = Array::<F, _>::zeros(lambda);
    let mut r2 = Array::<F, _>::zeros(lambda);
    r0.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });
    r1.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });
    r2.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });
    
    let mut H2 = Vec::<F>::new();
    H2.resize(code_len, zero);
    let all_zeros = Array::<F, _>::zeros(m);
    fill_H_array::<F>(
        &mut H2, n, m, lambda, &all_zeros, &v2, &w2, &r2
    );

    let mut H1 = Vec::<F>::new();
    H1.resize(code_len, zero);
    fill_H_array::<F>(
        &mut H1, n, m, lambda, &t, &v1, &w1, &r1
    );
        
    let mut H0 = Vec::<F>::new();
    H0.resize(code_len, zero);
    fill_H_array::<F>(
        &mut H0, n, m, lambda, &s, &v0, &w0, &r0
    );
    
    // encoding
    encode(&mut H2, &precodes, &postcodes);
    encode(&mut H1, &precodes, &postcodes);
    encode(&mut H0, &precodes, &postcodes);

    let hashes_E = merkle_tree_commit_lwe::<F, D>(code_len, &H2, &H1, &H0);


    // X
    let X = F::random(&mut rng);
    let X_invert = X.invert().unwrap();

    // fx: m
    let mut fx = Vec::<F>::new();
    fx.resize(m, zero);
    fx
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)|{
            *x = t[i].mul(X).add(s[i]);
        });
    let mut fx_copy = Vec::<F>::new();
    fx_copy.resize(m, zero);
    fx_copy
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)|{
            *x = t[i].mul(X).add(s[i]);
        });

    // rx: lambda
    let mut rx = Vec::<F>::new();
    rx.resize(lambda, zero);
    rx
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)|{
            *x = r2[i].mul(X).mul(X).add(
                r1[i].mul(X)
            ).add(
                r0[i]
            );
        });

    let committed_time = Instant::now();

    // verifier sampling idx
    let mut idx = Vec::<usize>::new();
    idx.resize_with(lambda, || rng.gen_range(0..code_len));

    let mut E_map = HashMap::<usize, Output<D>>::new();
    E_map.insert(0, hashes_E[0].clone());
    let E_map_rwlock = RwLock::new(E_map);

    (0..lambda).into_par_iter().for_each(|i| {
        // verify the merkle path for E
        let j = idx[i];
        let mut digest = D::new();
        digest.update(H2[j].to_repr());
        digest.update(H1[j].to_repr());
        digest.update(H0[j].to_repr());
        let cur_hash = digest.finalize();
        let item_no = code_len;
        let np2 = next_pow_2(item_no);
        let k = j + np2 - 1;
        assert!(check_merkle_path::<D>(cur_hash, k, &E_map_rwlock, &hashes_E));
    });

    let mut Afx = A.dot(&Array::from(fx_copy));
    let mut dx = Vec::<F>::new();
    dx.resize(n, zero);
    dx
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)|{
            *x = u[i].sub(Afx[i]);
        });

    // fxx: m
    let mut fxx = Vec::<F>::new();
    fxx.resize(m, zero);
    fxx
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)|{
            *x = fx[i].mul(
                fx[i].sub(one)
            ).mul(
                fx[i].add(one)
            ).mul(
                X_invert
            );
        });

    // dxx: n
    let mut dxx = Vec::<F>::new();
    dxx.resize(n, zero);
    dxx
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)|{
            *x = dx[i].mul(
                dx[i].sub(one)
            ).mul(
                dx[i].add(one)
            ).mul(
                X_invert
            );
        });


    let mut Hx = Vec::<F>::new();
    Hx.resize(code_len, zero);
    fill_H_array::<F>(
        &mut Hx, n, m, lambda, 
        &Array::from(fx), 
        &Array::from(fxx), 
        &Array::from(dxx), 
        &Array::from(rx)
    );
    
    // encoding
    encode(&mut Hx, &precodes, &postcodes);

    (0..lambda).into_par_iter().for_each(|i| {
        let j = idx[i];
        let x = H2[j].mul(X).mul(X).add(
            H1[j].mul(X)
        ).add(
            H0[j]
        );
        assert_eq!(Hx[j], x);
    });

    let verified_time = Instant::now();

    println!("n:{:?} m:{:?} lambda:{:?}", n, m, lambda);
    println!("commit_time: {} ms", committed_time.duration_since(start_time).as_millis());
    println!("verify_time: {} ms", verified_time.duration_since(committed_time).as_millis());
    println!("total_time: {} ms\n", verified_time.duration_since(start_time).as_millis());


}