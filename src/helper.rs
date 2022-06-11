use std::collections::HashMap;
use std::sync::RwLock;
use std::mem::drop;
use digest::Digest;
use digest::Output;
use ff::PrimeField;
use ndarray::Dim;
use ndarray::Array;
use ndarray::parallel::prelude::*;

pub fn next_pow_2(x: usize) -> usize {
    let mut y : usize = 1;
    while y < x {
        y *= 2;
    }
    return y;
}
