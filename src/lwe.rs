use ff::Field;
use ff::PrimeField;
use ndarray::Array;
use ndarray::Dim;
use ndarray::parallel::prelude::*;
use num_traits::Num;
use sprs::MulAcc;
use rand::Rng;


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


pub fn ternary_lwe<F>(
    n: usize,
    m: usize,
    lambda: usize,
)
where
    F: PrimeField + Num + MulAcc,
{
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
    for i in 0..n {
        u[i] = u[i].add(e[i]);
    }
    // u.par_iter_mut().zip(e.par_iter_mut()).for_each(|(x, y)| {
    //     *x = (*x).add(*y);
    // });


    // t: m
    let mut t = Array::<F, _>::zeros((m));
    t.par_iter_mut().for_each(|x| {
        let mut rng = rand::thread_rng();
        *x = F::random(&mut rng);
    });

    let mut v2 = Array::<F, _>::zeros((m));
    for i in 0..m {
        v2[i] = t[i].mul(t[i]).mul(t[i]);
    }

    let two = <F as Field>::one().add(<F as Field>::one());
    let mut v1 = Array::<F, _>::zeros((m));
    for i in 0..m {
        v1[i] = two.mul(s[i]).sub(<F as Field>::one()).mul(t[i]);
        v1[i] = v1[i].add(
            t[i].mul(t[i]).mul(
                s[i].add(<F as Field>::one())
            )
        );
    }

    let mut v0 = Array::<F, _>::zeros((m));
    for i in 0..m {
        v0[i] = t[i].mul(s[i]).mul(
            s[i].sub(<F as Field>::one())
        );
        v0[i] = v0[i].add(
            two.mul(s[i]).sub(<F as Field>::one()).mul(
                s[i].add(<F as Field>::one())
            )
        );
    }
    

    // At: n
    let mut At = A.dot(&t);

    let mut w2 = Array::<F, _>::zeros((m));
    for i in 0..n {
        w2[i] = <F as Field>::zero().sub(At[i].mul(At[i]).mul(At[i]));
    }

    let three = <F as Field>::one().add(<F as Field>::one()).add(<F as Field>::one());
    let mut w1 = Array::<F, _>::zeros((m));
    for i in 0..n {
        w1[i] = three.mul(e[i]).mul(At[i]).mul(At[i]);
    }

    let mut w0 = Array::<F, _>::zeros((m));
    for i in 0..n {
        w0[i] = <F as Field>::one().sub(
            three.mul(e[i]).mul(e[i])
        ).mul(At[i]);
    }

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
}