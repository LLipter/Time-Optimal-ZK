pub mod ft32 {
    use ff::PrimeField;
    use ff_derive_num::Num;

    #[derive(PrimeField, Num)]
    #[PrimeFieldModulus = "2147483647"]
    #[PrimeFieldGenerator = "3"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft32([u64; 1]);
}

pub mod ft127 {
    use ff::PrimeField;
    use ff_derive_num::Num;

    #[derive(PrimeField, Num)]
    #[PrimeFieldModulus = "146823888364060453008360742206866194433"]
    #[PrimeFieldGenerator = "3"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft127([u64; 2]);
}

pub mod ft255 {
    use ff::PrimeField;
    use ff_derive_num::Num;

    #[derive(PrimeField, Num)]
    #[PrimeFieldModulus = "46242760681095663677370860714659204618859642560429202607213929836750194081793"]
    #[PrimeFieldGenerator = "5"]
    #[PrimeFieldReprEndianness = "little"]
    pub struct Ft255([u64; 4]);
}
