mod codespec;
mod fieldspec;
mod codegen;
mod encode;
mod commit;
mod helper;
mod simple_zk;
mod merkle;

use fieldspec::ft127::Ft127;
use fieldspec::ft255::Ft255;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use ff::PrimeField;
use ff::Field;
use ndarray::Array;
use blake3::Hasher as Blake3;
use num_traits::pow;
use ndarray::Axis;
use ndarray::parallel::prelude::*;


fn main() {
    // rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    
    // commit::commit_2_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 1024, 1762, 0, 100);
    commit::commit_2_dim::<Ft255, codespec::Code6, Blake3>(10000, 100, 172, 0, 5);
    // commit::commit_3_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 101, 174, 0, 100);
    commit::commit_3_dim::<Ft255, codespec::Code6, Blake3>(27000, 30, 52, 0, 5);
    // commit::commit_4_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 32, 56, 0, 100);
    // commit::commit_4_dim::<Ft255, codespec::Code6, Blake3>(65536, 16, 28, 0, 5);

    // simple_zk::commit_2_dim_simple_zk::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 1024, 1762, 0, 100);
    simple_zk::commit_2_dim_simple_zk::<Ft255, codespec::Code6, Blake3>(10000, 100, 172, 0, 5);
    // simple_zk::commit_3_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 101, 174, 0, 100);
    // simple_zk::commit_3_dim_simple_zk::<Ft255, codespec::Code6, Blake3>(27000, 30, 52, 0, 5);
    // simple_zk::commit_4_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 32, 56, 0, 100);
    // simple_zk::commit_4_dim::<Ft255, codespec::Code6, Blake3>(65536, 16, 28, 0, 5);
}
