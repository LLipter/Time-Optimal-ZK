mod codespec;
mod fieldspec;
mod codegen;
mod encode;

use fieldspec::ft127::Ft127;
use fieldspec::ft255::Ft255;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use ff::PrimeField;
use ff::Field;



fn main() {
    println!("Hello, world!");
    let seed : u64 = 0;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let precode = codegen::gen_matrix::<Ft255, &mut ChaCha20Rng>(20, 10, 1, &mut rng);

    // println!("xxxx");
    // println!("{:?}", precode);

    println!("{}", codespec::binary_entropy(0.5));

    let (precodes, postcodes) = codegen::generate::<Ft255, codespec::Code6>(1000, 0);
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


    let mut data = Vec::<Ft255>::with_capacity(1720);
    for _ in 0..1000{
        data.push(Ft255::random(&mut rng));
    }
    for _ in 0..720{
        data.push(Ft255::zero());
    }
    // println!("{:?}", data);
    encode::encode(&mut data, &precodes, &postcodes);
    println!("encoded len: {}", data.len());
    // println!("{:?}", data);
}
