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

use ft127::Ft127;
use ft255::Ft255;
use sprs::CsMat;
use ff::Field;
use rand::Rng;
use rand::SeedableRng;
use rand::distributions::Uniform;
use rand_chacha::ChaCha20Rng;

// generate a code matrix of a given size with specified row density
fn gen_code<F, R>(n: usize, m: usize, d: usize, mut rng: R) -> CsMat<F>
where
    F: Field,
    R: Rng,
{
    let dist = Uniform::new(0, m);
    let mut data = Vec::<F>::with_capacity(d * n);
    let mut idxs = Vec::<usize>::with_capacity(d * n);
    let mut ptrs = Vec::<usize>::with_capacity(1 + n);
    ptrs.push(0); // ptrs always starts with 0

    for _ in 0..n {
        // for each row, sample d random nonzero columns (without replacement)
        let cols = {
            /*
            let mut nub = HashSet::new();
            let mut tmp = (&mut rng)
                .sample_iter(&dist)
                .filter(|&x| {
                    if nub.contains(&x) {
                        false
                    } else {
                        nub.insert(x);
                        true
                    }
                })
                .take(d)
                .collect::<Vec<_>>();
            */
            // for small d, the quadratic approach is almost certainly faster
            let mut tmp = Vec::with_capacity(d);
            assert_eq!(
                d,
                (&mut rng)
                    .sample_iter(&dist)
                    .filter(|&x| {
                        if tmp.contains(&x) {
                            false
                        } else {
                            tmp.push(x);
                            true
                        }
                    })
                    .take(d)
                    .count()
            );
            tmp.sort_unstable(); // need to sort to supply to new_csc below
            tmp
        };
        assert_eq!(d, cols.len());

        // sample random elements for each column
        let mut last = m + 1;
        for &col in &cols[..] {
            // detect and skip repeats
            if col == last {
                continue;
            }
            last = col;

            let val = {
                let mut tmp = F::random(&mut rng);
                while bool::from(<F as Field>::is_zero(&tmp)) {
                    tmp = F::random(&mut rng);
                }
                tmp
            };
            idxs.push(col);
            data.push(val);
        }
        ptrs.push(data.len());
    }

    CsMat::new_csc((m, n), ptrs, idxs, data)
}



fn main() {
    println!("Hello, world!");
    let seed : u64 = 0;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let precode = gen_code::<Ft255, &mut ChaCha20Rng>(20, 10, 1, &mut rng);

    // println!("xxxx");
    // println!("{:?}", precode);
}
