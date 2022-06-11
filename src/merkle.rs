use std::collections::HashMap;
use std::sync::RwLock;
use std::mem::drop;
use digest::Digest;
use digest::Output;
use ff::PrimeField;
use ndarray::Dim;
use ndarray::Array;
use ndarray::parallel::prelude::*;
use crate::helper::next_pow_2;

pub fn build_merkle_tree<D>(data: &mut Vec::<Output<D>>, np2: usize)
where
    D: Digest,
{
    for i in (0..(np2-1)).rev() {
        let mut digest = D::new();
        digest.update(&data[2*i+1]);
        digest.update(&data[2*i+2]);
        data[i] = digest.finalize();
    }
}

pub fn check_merkle_path<D>(
    mut cur_hash: Output<D>, 
    mut idx: usize, 
    hashes_map_rwlock: &RwLock<HashMap::<usize, Output<D>>>,
    hashes_vec: &Vec<Output<D>>
) -> bool 
where
    D: Digest
{
    while idx > 0 {
        let mut rwlock_write = hashes_map_rwlock.write().unwrap();
        match (*rwlock_write).get(&idx) {
            None => {
                (*rwlock_write).insert(idx, cur_hash.clone());
            },
            Some(h) => {
                if !cur_hash.eq(h) {
                    return false;
                }
            },
        }
        drop(rwlock_write);

        let mut digest = D::new();
        if idx % 2 == 0 {
            digest.update(&hashes_vec[idx-1]);
            digest.update(&cur_hash);
        }else{
            digest.update(&cur_hash);
            digest.update(&hashes_vec[idx+1]);
        }
        cur_hash = digest.finalize();
        idx = (idx - 1) / 2;
    }
    let rwlock_read = hashes_map_rwlock.read().unwrap();
    return cur_hash.eq(&((*rwlock_read)[&0]));
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