pub fn next_pow_2(x: usize) -> usize {
    let mut y : usize = 1;
    while y < x {
        y *= 2;
    }
    return y;
}