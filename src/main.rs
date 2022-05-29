mod codespec;
mod fieldspec;
mod codegen;
mod encode;
mod commit;
mod helper;

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
/*
    println!("Hello, world!");
    let seed : u64 = 0;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let precode = codegen::gen_matrix::<Ft255, &mut ChaCha20Rng>(20, 10, 1, &mut rng);

    // println!("xxxx");
    // println!("{:?}", precode);

    println!("{}", codespec::binary_entropy(0.5));

    let (precodes, postcodes) = codegen::generate::<Ft255, codespec::Code6>(100, 0);
    println!("{}", precodes.len());

    for pc in &precodes {
        println!("{}, {}", pc.cols(), pc.rows());
    }
    println!("xxxx");
    for pc in &postcodes {
        println!("{}, {}", pc.cols(), pc.rows());
    }
    println!("xxxx");
    println!("{}", encode::codeword_length(&precodes, &postcodes));


    let mut data = Vec::<Ft255>::with_capacity(172);
    for _ in 0..100{
        data.push(Ft255::random(&mut rng));
    }
    for _ in 0..72{
        data.push(Ft255::zero());
    }
    // println!("{:?}", data);
    encode::encode(&mut data, &precodes, &postcodes);
    println!("encoded len: {}", data.len());
    // println!("{:?}", data);

*/
    // let a = Array::<Ft255, _>::zeros((2, 3, 4));

    // // println!("{:?}", a);
    
    // let t = 5;
    // let mut vec = Vec::new();
    // for _ in 0..t{
    //     vec.push(t);
    // }
    // println!("{:?}", vec);
    // let b = Array::<Ft255, _>::zeros(vec);
    // // println!("{:?}", b);
    
    // let x = commit::random_coeffs_from_length::<Ft255>(10);
    // println!("{:?}", x);

    // let a = vec![1, 2, 3];
    // let b = vec![2, 3, 4];
    // let x = commit::get_1d_index(&a, &b);
    // println!("{}", x);

    // commit::commit_t_dim::<Ft255, codespec::Code6>(1000000, 3, 100, 172, 0);

    
    // commit::commit_2_dim::<Ft255, codespec::Code6, Blake3>(10000, 100, 172, 0, 5);
    // commit::commit_3_dim::<Ft255, codespec::Code6, Blake3>(1000000, 100, 172, 0, 5);
    
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    commit::commit_2_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 1024, 1762, 0, 100);
    commit::commit_2_dim::<Ft255, codespec::Code6, Blake3>(10000, 100, 172, 0, 5);
    commit::commit_3_dim::<Ft255, codespec::Code6, Blake3>(pow(2usize, 20), 101, 174, 0, 100);
    commit::commit_3_dim::<Ft255, codespec::Code6, Blake3>(27000, 30, 52, 0, 5);

}
