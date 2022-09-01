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
    H
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, mut x)|{
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


pub fn ternary_lwe<F, C>(
    n: usize,
    m: usize,
    lambda: usize,
    seed: u64,
)
where
    F: PrimeField + Num + MulAcc,
    C: CodeSpecification,
{
    let zero = <F as Field>::zero();
    let one = <F as Field>::one();
    let two = one.add(one);
    let three = two.add(one);

    // A: n * m
    let mut A = Array::<F, _>::zeros((n, m));
    A.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });

    // s: m
    let mut s = generate_ternary_vector::<F>(m);
    // e: n
    let mut e = generate_ternary_vector::<F>(n);

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


    // t: m
    let mut t = Array::<F, _>::zeros((m));
    t.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });

    let mut v2 = Array::<F, _>::zeros((m));
    v2
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = t[i].mul(t[i]).mul(t[i]);
            *x.first_mut().unwrap() = *data;
        });

    let mut v1 = Array::<F, _>::zeros((m));
    v1
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = two.mul(s[i]).sub(one).mul(t[i]);
            *data = data.add(
                t[i].mul(t[i]).mul(
                    s[i].add(one)
                )
            );
            *x.first_mut().unwrap() = *data;
        });

    let mut v0 = Array::<F, _>::zeros((m));
    v0
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = s[i].sub(one).mul(s[i]).mul(t[i]);
            *data = data.add(
                two.mul(s[i]).sub(one).mul(
                    s[i].add(one)
                )
            );
            *x.first_mut().unwrap() = *data;
        });

    

    // At: n
    let mut At = A.dot(&t);

    let mut w2 = Array::<F, _>::zeros((n));
    w2
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = zero.sub(At[i].mul(At[i]).mul(At[i]));
            *x.first_mut().unwrap() = *data;
        });

    let mut w1 = Array::<F, _>::zeros((n));
    w1
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut x)|{
            let data = x.first_mut().unwrap();
            *data = three.mul(e[i]).mul(At[i]).mul(At[i]);
            *x.first_mut().unwrap() = *data;
        });

    let mut w0 = Array::<F, _>::zeros((n));
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
    let mut r0 = Array::<F, _>::zeros((lambda));
    let mut r1 = Array::<F, _>::zeros((lambda));
    let mut r2 = Array::<F, _>::zeros((lambda));
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


    // generate codes
    let msg_len: usize = 2 * m + n + lambda;
    let (precodes, postcodes) = generate::<F, C>(msg_len, seed);
    let code_len = codeword_length::<F>(&precodes, &postcodes);
    
    let mut H2 = Vec::<F>::new();
    H2.resize(code_len, zero);
    let mut all_zeros = Array::<F, _>::zeros((m));
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

}