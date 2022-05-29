use std::collections::HashMap;
use std::sync::RwLock;
use std::mem::drop;
use digest::Digest;
use digest::Output;

pub fn next_pow_2(x: usize) -> usize {
    let mut y : usize = 1;
    while y < x {
        y *= 2;
    }
    return y;
}

pub fn build_merkle_tree<D>(data: &mut Vec::<Output<D>>, np2: usize)
where
    D: Digest,
{
    for i in (0..(np2-1)).rev() {
        // println!("{}", i);
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