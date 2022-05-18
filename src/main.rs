mod codespec;
mod fieldspec;
mod codegen;

use fieldspec::ft127::Ft127;
use fieldspec::ft255::Ft255;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;




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

    for pc in precodes {
        println!("{}, {}", pc.cols(), pc.rows());
    }
}
