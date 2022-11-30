mod codespec;
mod fieldspec;
mod codegen;
mod encode;
mod commit;
mod helper;
mod simple_zk;
mod merkle;
mod commit_zk;
mod lwe;

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
use fieldspec::ft32::Ft32;
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
use encode::encode_zk_bench;
use lwe::ternary_lwe;

fn main() {
    // rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();
    
    // commit::commit_2_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 1024, 1762, 0, 100);
    commit::commit_2_dim::<Ft255, codespec::Code6, Blake3>(10000, 100, 172, 0, 5);
    commit_zk::commit_2_dim_zk::<Ft255, codespec::Code6, Blake3>(10000, 100, 172, 0, 5);
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
    // commit_zk::commit_2_dim_zk::<Ft255, codespec::Code6, Blake3>(10000, 100, 172, 0, 5);
    // commit_zk::commit_3_dim_zk::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 101, 174, 0, 100);
    // commit_zk::commit_3_dim_zk::<Ft255, codespec::Code6, Blake3>(27000, 30, 52, 0, 5);
    // commit_zk::commit_4_dim_zk::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 32, 56, 0, 100);
    // commit_zk::commit_4_dim_zk::<Ft255, codespec::Code6, Blake3>(65536, 16, 28, 0, 5);

    // println!("{}", binary_entropy(0.5));
    // println!("{}", binary_entropy(0.1));
    // println!("{}", degree_bound(1.0/1.72, 256, 1762));
    // println!("{}", degree_bound(1.0/1.72, 256, 174));
    // println!("{}", degree_bound(1.0/1.72, 256, 56));

    // let (precodes, postcodes) = generate_rev::<Ft255, codespec::Code6>(1762, 0);
    // let mut data = Vec::<Ft255>::new();
    // let mut cur = <Ft255 as Field>::zero();
    // for i in 0..1762 {
    //     data.push(cur);
    //     cur = cur.add(<Ft255 as Field>::one());
    // }
    // // println!("{:?}", data);
    // encode_rev::<Ft255, _>(&mut data, &precodes, &postcodes);


    // // test reverse encoding
    // test_reverse_encoding::<Ft255, codespec::Code6>();

    // println!("{}", degree_bound(0.5, 256, 128));
    // println!("{}", degree_bound(0.45, 256, 128));
    // println!("{}", degree_bound(0.40, 256, 128));
    // println!("{}", degree_bound(0.35, 256, 128));
    // println!("{}", degree_bound(0.30, 256, 128));
    // println!("{}", degree_bound(0.25, 256, 128));
    // println!("{}", degree_bound(0.20, 256, 128));
    // println!("{}", degree_bound(0.15, 256, 128));
    // println!("{}", degree_bound(0.10, 256, 128));
    // println!("");

    // encode_zk_bench::<Ft255>(128, degree_bound(0.5, 256, 128));
    // encode_zk_bench::<Ft255>(128, degree_bound(0.45, 256, 128));
    // encode_zk_bench::<Ft255>(128, degree_bound(0.40, 256, 128));
    // encode_zk_bench::<Ft255>(128, degree_bound(0.35, 256, 128));
    // encode_zk_bench::<Ft255>(128, degree_bound(0.30, 256, 128));
    // encode_zk_bench::<Ft255>(128, degree_bound(0.25, 256, 128));
    // encode_zk_bench::<Ft255>(128, degree_bound(0.20, 256, 128));
    // encode_zk_bench::<Ft255>(128, degree_bound(0.15, 256, 128));
    // encode_zk_bench::<Ft255>(128, degree_bound(0.10, 256, 128));
    // encode_zk_bench::<Ft255>(128, degree_bound(0.32, 256, 128));
    // encode_zk_bench::<Ft255>(128, degree_bound(0.30, 256, 128));


    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 2048, 100, false);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(128, 128, 100, 0, false);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(256, 128, 100, 0, false);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(512, 128, 100, 0, false);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 128, 100, 0, false);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 256, 100, 0, false);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 512, 100, 0, false);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 1024, 100, 0, false);

    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(128, 128, 100, 0, true);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(256, 128, 100, 0, true);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(512, 128, 100, 0, true);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 128, 100, 0, true);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 256, 100, 0, true);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 512, 100, 0, true);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 1024, 100, 0, true);



    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 1024, 100, 0, false);
    // ternary_lwe::<Ft255, codespec::Code6, Blake3>(1024, 1024, 100, 0, false);

    // for i in (7..12){
    // for j in (7..12) {
    //         ternary_lwe::<Ft32, codespec::Code6, Blake3>(pow(2usize, i), pow(2usize, j), 200, 0, false);
    //     }
    // }

}