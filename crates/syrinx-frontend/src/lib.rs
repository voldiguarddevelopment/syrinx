//! syrinx-frontend — deterministic text frontend (T-00.01 scaffold; T-01.01
//! normalize; T-01.02 numeric expansion; T-01.04 G2P phonemization; T-01.05
//! custom pronunciation overrides; T-01.06 heteronym resolution; T-01.07 SSML
//! subset parsing).

pub mod expand;
pub mod g2p;
pub mod hetero;
pub mod lexicon;
pub mod normalize;
pub mod ssml;
