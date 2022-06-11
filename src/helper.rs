use ff::PrimeField;
use ndarray::Dim;
use ndarray::Array;
use ndarray::parallel::prelude::*;
use ndarray::Axis;
use num_traits::Num;

pub fn next_pow_2(x: usize) -> usize {
    let mut y : usize = 1;
    while y < x {
        y *= 2;
    }
    return y;
}

pub fn linear_combination_2_1<F>(
    msg_len: usize, 
    code_len: usize,
    m_2d: &Array<F, Dim<[usize; 2]>>,
    r: &Vec<F>,
    range: usize
) -> Array<F, Dim<[usize; 1]>>
where
    F: PrimeField + Num,
{
    assert_eq!(m_2d.shape(), &[code_len, msg_len]);
    assert_eq!(r.len(), msg_len);
    assert!(range == msg_len || range == code_len);

    let mut result = Array::<F, _>::zeros((range));
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
    r: &Vec<F>,
    range: usize
) -> Array<F, Dim<[usize; 2]>>
where
    F: PrimeField + Num,
{
    assert_eq!(m_3d.shape(), &[code_len, code_len, msg_len]);
    assert_eq!(r.len(), msg_len);
    assert!(range == msg_len || range == code_len);

    let mut result = Array::<F, _>::zeros((code_len, range));
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
    r: &Vec<F>,
    range: usize
) -> Array<F, Dim<[usize; 3]>>
where
    F: PrimeField + Num,
{
    assert_eq!(m_4d.shape(), &[code_len, code_len, code_len, msg_len]);
    assert_eq!(r.len(), msg_len);
    assert!(range == msg_len || range == code_len);

    let mut result = Array::<F, _>::zeros((code_len, code_len, range));
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