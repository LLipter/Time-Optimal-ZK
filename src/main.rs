pub mod ft127 {
    use ff::PrimeField;

    #[derive(PrimeField)]
    #[PrimeFieldModulus = "146823888364060453008360742206866194433"]
    #[PrimeFieldGenerator = "3"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft127([u64; 2]);
}

pub mod ft255 {
    use ff::PrimeField;

    #[derive(PrimeField)]
    #[PrimeFieldModulus = "46242760681095663677370860714659204618859642560429202607213929836750194081793"]
    #[PrimeFieldGenerator = "5"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft255([u64; 4]);
}


fn main() {
    println!("Hello, world!");
}
