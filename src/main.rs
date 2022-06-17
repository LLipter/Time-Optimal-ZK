mod codespec;
mod fieldspec;
mod codegen;
mod encode;
mod commit;
mod helper;
mod simple_zk;
mod merkle;
mod commit_zk;

use std::collections::HashMap;
use std::time::Instant;
use std::sync::RwLock;
use rand::Rng;
use ff::Field;
use ff::PrimeField;
use ndarray::Array;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::parallel::prelude::*;
use num_traits::Num;
use sprs::MulAcc;
use digest::Digest;
use digest::Output;
use sprs::CsMat;
use fieldspec::ft127::Ft127;
use fieldspec::ft255::Ft255;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use blake3::Hasher as Blake3;
use num_traits::pow;
use ndarray::parallel::prelude::*;
use helper::binary_entropy;
use helper::degree_bound;
use codegen::generate_rev;
use std::ops::Add;
use encode::encode_rev;
use encode::test_reverse_encoding;


fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    
    // commit::commit_2_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 1024, 1762, 0, 100);
    // commit::commit_2_dim::<Ft255, codespec::Code6, Blake3>(10000, 100, 172, 0, 5);
    // commit::commit_3_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 101, 174, 0, 100);
    // commit::commit_3_dim::<Ft255, codespec::Code6, Blake3>(27000, 30, 52, 0, 5);
    // commit::commit_4_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 32, 56, 0, 100);
    // commit::commit_4_dim::<Ft255, codespec::Code6, Blake3>(65536, 16, 28, 0, 5);
    
    // simple_zk::commit_2_dim_simple_zk::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 1024, 1762, 0, 100);
    // simple_zk::commit_2_dim_simple_zk::<Ft255, codespec::Code6, Blake3>(10000, 100, 172, 0, 5);
    // simple_zk::commit_3_dim_simple_zk::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 101, 174, 0, 100);
    // simple_zk::commit_3_dim_simple_zk::<Ft255, codespec::Code6, Blake3>(27000, 30, 52, 0, 5);
    // simple_zk::commit_4_dim_simple_zk::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 32, 56, 0, 100);
    // simple_zk::commit_4_dim_simple_zk::<Ft255, codespec::Code6, Blake3>(65536, 16, 28, 0, 5);

    // commit_zk::commit_2_dim_zk::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 1024, 1762, 0, 100);
    commit_zk::commit_2_dim_zk::<Ft255, codespec::Code6, Blake3>(10000, 100, 172, 0, 5);
    // commit_zk::commit_3_dim_zk::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 101, 174, 0, 100);
    // commit_zk::commit_3_dim_zk::<Ft255, codespec::Code6, Blake3>(27000, 30, 52, 0, 5);
    // commit_zk::commit_4_dim_zk::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 32, 56, 0, 100);
    // commit_zk::commit_4_dim_zk::<Ft255, codespec::Code6, Blake3>(65536, 16, 28, 0, 5);

    println!("{}", binary_entropy(0.5));
    println!("{}", binary_entropy(0.1));
    println!("{}", degree_bound(1.0/1.72, 256, 1762));
    println!("{}", degree_bound(1.0/1.72, 256, 174));
    println!("{}", degree_bound(1.0/1.72, 256, 56));

    let (precodes, postcodes) = generate_rev::<Ft255, codespec::Code6>(1762, 0);
    let mut data = Vec::<Ft255>::new();
    let mut cur = <Ft255 as Field>::zero();
    for i in 0..1762 {
        data.push(cur);
        cur = cur.add(<Ft255 as Field>::one());
    }
    // println!("{:?}", data);
    encode_rev::<Ft255, _>(&mut data, &precodes, &postcodes);


    // TODO: test reverse encoding
    test_reverse_encoding::<Ft255, codespec::Code6>();
}
