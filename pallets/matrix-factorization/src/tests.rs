use super::RawEvent;
use crate::{Error, Module, Trait};
use frame_support::{assert_err, assert_ok, impl_outer_event, impl_outer_origin, parameter_types};
use frame_system as system;
use sp_core::H256;
use sp_io::TestExternalities;
use sp_runtime::{
	testing::Header,
	traits::{BlakeTwo256, IdentityLookup},
	Perbill,
};

impl_outer_origin! {
	pub enum Origin for TestRuntime {}
}

// Workaround for https://github.com/rust-lang/rust/issues/26925 . Remove when sorted.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TestRuntime;
parameter_types! {
	pub const BlockHashCount: u64 = 250;
	pub const MaximumBlockWeight: u32 = 1024;
	pub const MaximumBlockLength: u32 = 2 * 1024;
	pub const AvailableBlockRatio: Perbill = Perbill::one();
}
impl system::Trait for TestRuntime {
	type Origin = Origin;
	type Index = u64;
	type Call = ();
	type BlockNumber = u64;
	type Hash = H256;
	type Hashing = BlakeTwo256;
	type AccountId = u64;
	type Lookup = IdentityLookup<Self::AccountId>;
	type Header = Header;
	type Event = TestEvent;
	type BlockHashCount = BlockHashCount;
	type MaximumBlockWeight = MaximumBlockWeight;
	type DbWeight = ();
	type BlockExecutionWeight = ();
	type ExtrinsicBaseWeight = ();
	type MaximumExtrinsicWeight = MaximumBlockWeight;
	type MaximumBlockLength = MaximumBlockLength;
	type AvailableBlockRatio = AvailableBlockRatio;
	type Version = ();
	type ModuleToIndex = ();
	type AccountData = ();
	type OnNewAccount = ();
	type OnKilledAccount = ();
}

mod matrix_factorization {
	pub use crate::Event;
}

impl_outer_event! {
	pub enum TestEvent for TestRuntime {
		matrix_factorization<T>,
		system<T>,
	}
}

impl Trait for TestRuntime {
	type Event = TestEvent;
	const NUM_ITEMS: usize = 16;
	const EMBEDDING_DIM: usize = 64;
	const FRACTIONAL_BITS: u32 = 24;
	const INITIAL_RND_AMPLITUDE: i64 = 1 << 10;
	const PRIOR_PRECISION: i64 = 1 << 24;
	const LIKELIHOOD_PRECISION: i64 = 3 << 23;
}

pub type System = system::Module<TestRuntime>;
pub type MatrixFactorization = Module<TestRuntime>;

pub struct ExtBuilder;

impl ExtBuilder {
	pub fn build() -> TestExternalities {
		let storage = system::GenesisConfig::default()
			.build_storage::<TestRuntime>()
			.unwrap();
		let mut ext = TestExternalities::from(storage);
		ext.execute_with(|| System::set_block_number(1));
		ext
	}
}

#[test]
fn initial_prediction() {
	ExtBuilder::build().execute_with(|| {
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 5));

		let raw_event = System::events().iter().find_map(|a| {
			if let TestEvent::matrix_factorization(RawEvent::PredictionObtained(
				account,
				item,
				prediction,
			)) = a.event
			{
				Some((account, item, prediction))
			} else {
				None
			}
		});

		let (account, item, prediction) = raw_event.unwrap();
		assert_eq!(account, 1);
		assert_eq!(item, 5);
		assert_eq!(prediction, 0);
	})
}

#[test]
fn test_lin_solve() {
	ExtBuilder::build().execute_with(|| {
		let K = TestRuntime::EMBEDDING_DIM as usize;

		// [2,  3, 0, 0, ...]
		// [1, -2, 0, 0, ...]
		// [0,  0, 1, 0, ...]
		// [0,  0, 0, 1, ...]
		let mut a = vec![0i64; K * K];
		a[0] = 2 << 24;
		a[1] = 3 << 24;
		a[K] = 1 << 24;
		a[K + 1] = -2 << 24;
		for i in 2..K {
			a[(K + 1) * i] = 1 << 24;
		}

		use rand_xoshiro::{
			rand_core::{RngCore, SeedableRng},
			Xoshiro256PlusPlus,
		};
		let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
		for ai in a.iter_mut() {
			*ai = ((rng.next_u32() & 0x0f) as i64 - 8) << 24
		}

		let mut x = vec![0i64; K];
		x[0] = 2 << 24;
		x[1] = 3 << 24;

		let mut x2 = vec![0i64; K];
		x2[0] = 2;
		x2[1] = 3;

		let mut y = vec![0i64; K];
		y[0] = 13 << 24;
		y[1] = -4 << 24;

		let sol = MatrixFactorization::lin_solve(a, &y);
		assert_eq!(sol.clone(), x.clone());
		dbg!(sol.clone());
		dbg!(x);

		let sol2 = sol.iter().map(|x| *x >> 24).collect::<Vec<_>>();
		assert_eq!(sol2, x2);
	})
}

#[test]
fn test_lin_solve2() {
	use rand_xoshiro::{
		rand_core::{RngCore, SeedableRng},
		Xoshiro256PlusPlus,
	};

	fn rnd_int_vec(seed: u64, shift: u32, len: usize) -> Vec<i64> {
		let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
		(0..len)
			.map(|_| ((rng.next_u32() & 0x0f) as i64 - 8) << (TestRuntime::FRACTIONAL_BITS - shift))
			.collect()
	}

	ExtBuilder::build().execute_with(|| {
		let K = TestRuntime::EMBEDDING_DIM as usize;

		let mut a = rnd_int_vec(123, 15, K * K);
		for a_diag in a.iter_mut().step_by(K + 1) {
			*a_diag += 1 << TestRuntime::FRACTIONAL_BITS;
		}

		let mut x = rnd_int_vec(1234, 4, K);
		let y = a
			.chunks(K)
			.map(|row| {
				row.iter().zip(&x).map(|(ai, xi)| ai * xi).sum::<i64>()
					>> TestRuntime::FRACTIONAL_BITS
			})
			.collect::<Vec<i64>>();

		let mut sol = MatrixFactorization::lin_solve(a, &y);

		for sol_i in sol.iter_mut() {
			*sol_i >>= 21
		}
		for x_i in x.iter_mut() {
			*x_i >>= 21
		}
		assert_eq!(&sol, &x);
		assert!(&sol != &y);
	})
}

#[test]
fn test_optimization() {
	ExtBuilder::build().execute_with(|| {
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 0));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 1));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(2), 0));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(2), 1));

		assert_ok!(MatrixFactorization::submit_rating(Origin::signed(1), 0, 1));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 0));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 1));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(2), 0));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(2), 1));

		assert_ok!(MatrixFactorization::submit_rating(Origin::signed(2), 0, 1));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 0));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 1));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(2), 0));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(2), 1));

		assert_ok!(MatrixFactorization::submit_rating(Origin::signed(2), 1, 1));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 0));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 1));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(2), 0));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(2), 1));

		assert_ok!(MatrixFactorization::submit_rating(Origin::signed(1), 1, 1));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 0));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(1), 1));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(2), 0));
		assert_ok!(MatrixFactorization::get_prediction(Origin::signed(2), 1));
	})
}

// #[test]
// fn set_works() {
// 	ExtBuilder::build().execute_with(|| {
// 		assert_ok!(MatrixFactorization::set_single_entry(Origin::signed(1), 19));

// 		let expected_event = TestEvent::matrix_factorization(RawEvent::EntrySet(1, 19));

// 		assert!(System::events().iter().any(|a| a.event == expected_event));
// 	})
// }

// #[test]
// fn get_throws() {
// 	ExtBuilder::build().execute_with(|| {
// 		assert_err!(
// 			MatrixFactorization::get_single_entry(Origin::signed(2), 3),
// 			Error::<TestRuntime>::NoValueStored
// 		);
// 	})
// }

// #[test]
// fn get_works() {
// 	ExtBuilder::build().execute_with(|| {
// 		assert_ok!(MatrixFactorization::set_single_entry(Origin::signed(2), 19));
// 		assert_ok!(MatrixFactorization::get_single_entry(Origin::signed(1), 2));

// 		let expected_event = TestEvent::matrix_factorization(RawEvent::EntryGot(1, 19));
// 		assert!(System::events().iter().any(|a| a.event == expected_event));

// 		// Ensure storage is still set
// 		assert_eq!(MatrixFactorization::matrix_factorization(2), 19);
// 	})
// }

// #[test]
// fn take_throws() {
// 	ExtBuilder::build().execute_with(|| {
// 		assert_err!(
// 			MatrixFactorization::take_single_entry(Origin::signed(2)),
// 			Error::<TestRuntime>::NoValueStored
// 		);
// 	})
// }

// #[test]
// fn take_works() {
// 	ExtBuilder::build().execute_with(|| {
// 		assert_ok!(MatrixFactorization::set_single_entry(Origin::signed(2), 19));
// 		assert_ok!(MatrixFactorization::take_single_entry(Origin::signed(2)));

// 		let expected_event = TestEvent::matrix_factorization(RawEvent::EntryTaken(2, 19));
// 		assert!(System::events().iter().any(|a| a.event == expected_event));

// 		// Assert storage has returned to default value (zero)
// 		assert_eq!(MatrixFactorization::matrix_factorization(2), 0);
// 	})
// }

// #[test]
// fn increase_works() {
// 	ExtBuilder::build().execute_with(|| {
// 		assert_ok!(MatrixFactorization::set_single_entry(Origin::signed(2), 19));
// 		assert_ok!(MatrixFactorization::increase_single_entry(
// 			Origin::signed(2),
// 			2
// 		));

// 		let expected_event = TestEvent::matrix_factorization(RawEvent::EntryIncreased(2, 19, 21));

// 		assert!(System::events().iter().any(|a| a.event == expected_event));
// 	})
// }
