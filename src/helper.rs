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