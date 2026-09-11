//! Which geometries reproduce the C4.2′ pattern `p(cue vs sham) << p(cue vs plain)`?
//!
//! The 2026-09-09 certification run (`renders/2026-09-09-c42-certification/`) found two
//! cases whose cue separated from its sham at the permutation test's resolution floor while
//! separating from plain only moderately, and FINDINGS.md read that as the cue and the sham
//! moving in **different directions** from plain. This example is the adversarial check on
//! that reading, and it is what the ADR-0003 amendment's tables were produced with — so the
//! numbers in that ADR are reproducible from disk rather than asserted.
//!
//! It drives the REAL statistic — `acoustic::activation_test` — over synthetic arms whose
//! geometry is known, so nothing here can drift from the implementation it is reasoning
//! about. Five configurations, all at n=9 in 11 dimensions to match the run:
//!
//!   collinear     cue and sham displaced the same way   <- the null for "different directions"
//!   orthogonal    displaced 90 deg apart, no opposition
//!   opposed       displaced 180 deg apart               <- what the recorded hypothesis claims
//!   variance      IDENTICAL instructed means, plain arm wider
//!   two shams     two DELIVERY-NEUTRAL arms, orthogonal <- no content at all
//!
//! Reading it: if `orthogonal` and `opposed` give the same signature, the run's p-values
//! cannot tell them apart. If `two shams` gives that signature too, `cue vs sham` alone
//! does not measure delivery content.
//!
//!     cargo run --release -p syrinx-eval --example contrast_geometry [reps]

use syrinx_eval::acoustic::{activation_test, Features, N_FEATURES};

const N: usize = 9;

/// xorshift64* — a self-contained generator, so the table is reproducible with no dep.
struct Rng(u64);
impl Rng {
    fn bits(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f64 {
        (self.bits() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-15);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn arm(rng: &mut Rng, mu: &[f64; N_FEATURES], sd: f64) -> Vec<Features> {
    (0..N)
        .map(|_| {
            let mut v = [0.0f64; N_FEATURES];
            for d in 0..N_FEATURES {
                v[d] = mu[d] + sd * rng.normal();
            }
            Features { v }
        })
        .collect()
}

/// alpha = 1.0: the test must never decline to report a p-value here. This example reads
/// `p_value`, never `activated` — it is measuring the statistic's geometry, not gating.
fn p(a: &[Features], b: &[Features]) -> f64 {
    activation_test(a, b, 1.0).expect("equal, non-empty groups").p_value
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN p-values"));
    v[v.len() / 2]
}

fn axis(i: usize, m: f64) -> [f64; N_FEATURES] {
    let mut v = [0.0f64; N_FEATURES];
    v[i] = m;
    v
}

fn row(
    label: &str,
    mu_cue: [f64; N_FEATURES],
    mu_sham: [f64; N_FEATURES],
    sd_plain: f64,
    sd_instructed: f64,
    reps: usize,
    rng: &mut Rng,
) {
    let zero = [0.0f64; N_FEATURES];
    let (mut cp, mut sp, mut cs) = (Vec::new(), Vec::new(), Vec::new());
    let mut smaller = 0usize;
    for _ in 0..reps {
        let plain = arm(rng, &zero, sd_plain);
        let cue = arm(rng, &mu_cue, sd_instructed);
        let sham = arm(rng, &mu_sham, sd_instructed);
        let (a, b, c) = (p(&cue, &plain), p(&sham, &plain), p(&cue, &sham));
        if c < a {
            smaller += 1;
        }
        cp.push(a);
        sp.push(b);
        cs.push(c);
    }
    println!(
        "  {:<40} {:>10.4} {:>11.4} {:>10.5} {:>10.2}",
        label,
        median(&mut cp),
        median(&mut sp),
        median(&mut cs),
        smaller as f64 / reps as f64
    );
}

fn main() {
    let reps: usize =
        std::env::args().nth(1).and_then(|a| a.parse().ok()).filter(|r| *r > 0).unwrap_or(25);
    let mut rng = Rng(0x5EED_C42D_0003_0001);

    println!(
        "C4.2' contrast geometry — n={N}/arm, {N_FEATURES} dims, exact permutation test, \
         {reps} draws/cell (medians)\n"
    );
    println!("  {:<40} {:>10} {:>11} {:>10} {:>10}", "", "cue/plain", "sham/plain", "cue/sham", "P(cs<cp)");

    println!("\n-- displacement DIRECTION (equal variance, equal magnitude) --");
    for m in [0.8, 1.2, 1.6] {
        row(&format!("collinear (same direction), |mu|={m}"), axis(0, m), axis(0, m), 1.0, 1.0, reps, &mut rng);
    }
    for m in [0.8, 1.2, 1.6] {
        row(&format!("orthogonal (90 deg), |mu|={m}"), axis(0, m), axis(1, m), 1.0, 1.0, reps, &mut rng);
    }
    for m in [0.8, 1.2, 1.6] {
        row(&format!("opposed (180 deg), |mu|={m}"), axis(0, m), axis(0, -m), 1.0, 1.0, reps, &mut rng);
    }

    println!("\n-- UNEQUAL VARIANCE alone: cue and sham share one mean, plain is wider --");
    for k in [1.0, 1.6, 2.0] {
        row(&format!("identical instructed means, sd(plain)={k}x"), axis(0, 1.2), axis(0, 1.2), k, 1.0, reps, &mut rng);
    }

    println!("\n-- THE MISSING ARM: two DELIVERY-NEUTRAL shams, orthogonal, zero content --");
    for m in [0.8, 1.2, 1.6] {
        row(&format!("sham1 vs sham2, |mu|={m}"), axis(0, m), axis(1, m), 1.0, 1.0, reps, &mut rng);
    }

    println!(
        "\nRead: only `collinear` fails to produce the pattern, `orthogonal` and `opposed` are\n\
         indistinguishable, unequal variance alone does not produce it, and two meaningless\n\
         instructions produce it in full. See the ADR-0003 amendment."
    );
}
