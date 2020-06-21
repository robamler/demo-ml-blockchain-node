//! A demo decentralized recommendation system using matrix factorization

#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::{
	debug, decl_error, decl_event, decl_module, decl_storage,
	dispatch::{DispatchResult, Vec},
};
use frame_system::{self as system, ensure_signed};
use rand_xoshiro::{
	rand_core::{RngCore, SeedableRng},
	Xoshiro256PlusPlus,
};

#[cfg(test)]
mod tests;

pub trait Trait: system::Trait {
	type Event: From<Event<Self>> + Into<<Self as system::Trait>::Event>;
	const NUM_ITEMS: usize;
	const EMBEDDING_DIM: usize;
	const FRACTIONAL_BITS: u32;
	const INITIAL_RND_AMPLITUDE: i64;
	const PRIOR_PRECISION: i64;
	const LIKELIHOOD_PRECISION: i64;
}

// #[derive(Debug, Encode, Decode, Clone, PartialEq, Eq)]
// pub enum Rating {
// 	Like,
// 	Dislike,
// }

decl_storage! {
	trait Store for Module<T: Trait> as MatrixFactorization {
		QItem get(fn get_raw_q_item): Option<(Vec<i64>, Vec<i64>)>;
		QUser: map hasher(blake2_128_concat) T::AccountId => (Vec<i64>, Vec<i64>, Vec<i64>);
	}
}

decl_event!(
	pub enum Event<T>
	where
		AccountId = <T as system::Trait>::AccountId,
	{
		/// A user has submitted a rating.
		RatingSubmitted(AccountId, u64, i32),

		/// The model has calculated a prediction.
		PredictionObtained(AccountId, u64, i64),
	}
);

decl_error! {
	pub enum Error for Module<T: Trait> {
		/// The rated or requested item does not exist.
		ItemDoesNotExist,
	}
}

// impl From<Rating> for i64 {
// 	fn from(rating: Rating) -> i64 {
// 		match rating {
// 			Rating::Like => 1,
// 			Rating::Dislike => -1,
// 		}
// 	}
// }

decl_module! {
	pub struct Module<T: Trait> for enum Call where origin: T::Origin {

		// Initialize errors
		type Error = Error<T>;

		// Initialize events
		fn deposit_event() = default;

		/// Set the value stored at a particular key
		#[weight = 10_000]
		fn submit_rating(origin, item: u64, rating: i64) -> DispatchResult {
			let user = ensure_signed(origin)?;

			let (mut natural_parameter, mut precision_uu, mut offdiagonal_precisions) = Self::get_q_user(&user);
			let (mut item_means, mut item_precisions) = Self::get_q_item();

			let mean_i = item_means[item as usize..].iter().step_by(T::NUM_ITEMS as usize).cloned().collect::<Vec<i64>>();
			let precision_ii = item_precisions[item as usize..].iter().step_by(T::NUM_ITEMS as usize).cloned().collect::<Vec<i64>>();
			let precision_ui = offdiagonal_precisions[item as usize..].iter().step_by(T::NUM_ITEMS as usize).cloned().collect::<Vec<i64>>();

			let (opt_item_i, opt_user_j, other_correlations) = Self::coordinate_ascent(
				&precision_ii,
				&mean_i,
				&offdiagonal_precisions,
				&item_means,
				&precision_uu,
				&precision_ui,
				&natural_parameter,
				rating,
			);

			for (target, source) in item_means[item as usize..]
				.iter_mut()
				.step_by(T::NUM_ITEMS as usize)
				.zip(&opt_item_i)
			{
				*target = *source;
			}

			for (target, source) in item_precisions[item as usize..]
				.iter_mut()
				.step_by(T::NUM_ITEMS as usize)
				.zip(&opt_user_j)
			{
				*target += (
					(T::LIKELIHOOD_PRECISION * *source) >> T::FRACTIONAL_BITS) * *source
					>> T::FRACTIONAL_BITS;
			}

			let const_part = (
				T::LIKELIHOOD_PRECISION * (
					Self::dot(&opt_item_i,&opt_user_j) - (rating << T::FRACTIONAL_BITS))
				) >> T::FRACTIONAL_BITS;
			for ((target, source1), source2) in offdiagonal_precisions[item as usize..]
				.iter_mut()
				.step_by(T::NUM_ITEMS as usize)
				.zip(&opt_item_i)
				.zip(&opt_user_j)
			{
				*target += const_part + (
					(T::LIKELIHOOD_PRECISION * *source1) >> T::FRACTIONAL_BITS) * *source2
					>> T::FRACTIONAL_BITS;
			}

			for (((((target, precision_uu1), opt_user_j1), precision_ui1), opt_item_i1), other_correlations1) in natural_parameter
				.iter_mut()
				.zip(&precision_uu)
				.zip(&opt_user_j)
				.zip(&precision_ui)
				.zip(&opt_item_i)
				.zip(&other_correlations)
			{
				*target += ((precision_uu1 * opt_user_j1 + precision_ui1 * opt_item_i1) >> T::FRACTIONAL_BITS) + other_correlations1;
			}

			Self::set_q_user(&user, natural_parameter, precision_uu, offdiagonal_precisions);
			Self::set_q_item(item_means, item_precisions);


			// 	precision_uu[user, :] += LIKELIHOOD_PRECISION * opt_item_i**2
			// 	nu_u[user, :] = (
			// 		precision_uu[user, :] * opt_user_j + other_correlations + precision_ui[user, item, :] * mu_i[item, :])			// <MatrixFactorization<T>>::insert(&user, entry);

			// Self::deposit_event(RawEvent::EntrySet(user, entry));
			Ok(())
		}

		/// Set the value stored at a particular key
		#[weight = 10_000]
		fn get_prediction(origin, item: u64) -> DispatchResult {
			let user = ensure_signed(origin)?;

			let (natural_parameter, precision_uu, offdiagonal_precisions) = Self::get_q_user(&user);
			let (item_means, _) = Self::get_q_item();

			let mean_i = item_means[item as usize..].iter().step_by(T::NUM_ITEMS as usize);
			let precision_times_mu = Self::marginalize_last_dim(&offdiagonal_precisions, &item_means);
			let prediction = natural_parameter
				.iter()
				.zip(precision_times_mu)
				.zip(precision_uu)
				.zip(mean_i)
				.map(
					|(((nu, precmu), prec), mean_i)| (nu - precmu) * mean_i / prec
				).sum();

			dbg!((&user, item, prediction));
			// def prediction(item, user):
			// 	   mu_j = (nu_u[user, :] - np.sum(offdiagonal_precisions[user, :, :] * mu_i, axis=0)) / precision_uu[user, :]
			// 	   return mu_i[item, :].dot(mu_j)
			Self::deposit_event(RawEvent::PredictionObtained(user, item, prediction));
			Ok(())
		}
	}
}

impl<T: Trait> Module<T> {
	fn get_q_item() -> (Vec<i64>, Vec<i64>) {
		if let Some(raw_q_item) = Self::get_raw_q_item() {
			raw_q_item
		} else {
			let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
			let len = (T::EMBEDDING_DIM * T::NUM_ITEMS) as usize;
			let mut item_means = Vec::with_capacity(len);
			item_means.resize_with(len, || {
				let rnd = rng.next_u32() as i64 - 0x8000_0000;
				(rnd * T::INITIAL_RND_AMPLITUDE) >> 31
			});

			let item_precisions = vec![T::PRIOR_PRECISION; len];

			(item_means, item_precisions)
		}
	}

	fn set_q_item(item_means: Vec<i64>, item_precisions: Vec<i64>) {
		<Module<T> as Store>::QItem::put((item_means, item_precisions))
	}

	fn get_q_user(user: &T::AccountId) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
		if <Module<T> as Store>::QUser::contains_key(user) {
			<Module<T> as Store>::QUser::get(user)
		} else {
			let natural_parameter = vec![0i64; T::EMBEDDING_DIM as usize];
			let precision_uu = vec![T::PRIOR_PRECISION; T::EMBEDDING_DIM as usize];
			let offdiagonal_precisions = vec![0i64; (T::EMBEDDING_DIM * T::NUM_ITEMS) as usize];

			(natural_parameter, precision_uu, offdiagonal_precisions)
		}
	}

	fn set_q_user(
		user: &T::AccountId,
		natural_parameter: Vec<i64>,
		precision_uu: Vec<i64>,
		offdiagonal_precisions: Vec<i64>,
	) {
		<Module<T> as Store>::QUser::insert(
			user,
			(natural_parameter, precision_uu, offdiagonal_precisions),
		)
	}

	fn marginalize_last_dim<'a>(
		mat1: &'a [i64],
		mat2: &'a [i64],
	) -> impl Iterator<Item = i64> + 'a {
		assert_eq!(mat1.len(), (T::EMBEDDING_DIM * T::NUM_ITEMS) as usize);
		assert_eq!(mat1.len(), mat2.len());

		mat1.chunks(T::EMBEDDING_DIM as usize)
			.zip(mat2.chunks(T::EMBEDDING_DIM as usize))
			.map(|(vec1, vec2)| {
				vec1.iter().zip(vec2).map(|(a, b)| a * b).sum::<i64>() >> T::FRACTIONAL_BITS
			})
	}

	fn dot(a: &[i64], b: &[i64]) -> i64 {
		a.iter().zip(b).map(|(a_i, b_i)| a_i * b_i).sum::<i64>() >> T::FRACTIONAL_BITS
	}

	fn coordinate_ascent(
		precision_ii: &[i64],
		mean_i: &[i64],
		offdiagonal_precisions: &[i64],
		item_means: &[i64],
		precision_uu: &[i64],
		precision_ui: &[i64],
		natural_parameter: &[i64],
		rating: i64,
	) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
		let bias_i = Self::dot(precision_ii, mean_i);

		let precision_rating = T::LIKELIHOOD_PRECISION * rating;
		let all_correlations =
			Self::marginalize_last_dim(offdiagonal_precisions, item_means).collect::<Vec<i64>>();
		let own_correlations = precision_ui
			.iter()
			.zip(mean_i)
			.map(|(a, b)| (a * b) >> T::FRACTIONAL_BITS);
		let other_correlations = all_correlations
			.iter()
			.zip(own_correlations)
			.map(|(a, b)| a - b)
			.collect::<Vec<i64>>();

		// Initialization.
		let mut opt_item_i = Vec::from(mean_i);

		let mut opt_user_j = natural_parameter
			.iter()
			.zip(&all_correlations)
			.zip(precision_uu)
			.map(|((nu, precmu), prec)| ((nu - precmu) << T::FRACTIONAL_BITS) / prec)
			.collect::<Vec<i64>>();

		let mut mat = vec![0i64; (T::EMBEDDING_DIM * T::EMBEDDING_DIM) as usize];
		let mut vec = vec![0i64; T::EMBEDDING_DIM as usize];

		for iteration in 0..30 {
			let prediction = Self::dot(&opt_user_j, &opt_item_i);
			dbg!((iteration, prediction));

			// Solve for opt_user_j
			let mut mat_iter = mat.iter_mut();
			for opt_item_i1 in opt_item_i.iter() {
				for (opt_item_i2, target) in opt_item_i.iter().zip(&mut mat_iter) {
					*target = (((T::LIKELIHOOD_PRECISION * opt_item_i1) >> T::FRACTIONAL_BITS)
						* opt_item_i2) >> T::FRACTIONAL_BITS;
				}
			}
			assert!(mat_iter.next().is_none());

			for (source, target) in precision_uu
				.iter()
				.zip(mat.iter_mut().step_by(T::EMBEDDING_DIM as usize + 1))
			{
				*target += source
			}

			for (
				(((target, precision_ui1), opt_item_i1), natural_parameter1),
				other_correlations1,
			) in vec
				.iter_mut()
				.zip(precision_ui)
				.zip(&opt_item_i)
				.zip(natural_parameter)
				.zip(&other_correlations)
			{
				*target = (((precision_rating - precision_ui1) * opt_item_i1) >> T::FRACTIONAL_BITS)
					+ natural_parameter1 - other_correlations1
			}

			std::mem::replace(&mut opt_user_j, Self::lin_solve(mat.clone(), &vec));

			// Solve for opt_item_i
			let mut mat_iter = mat.iter_mut();
			for opt_user_j1 in opt_user_j.iter() {
				for (opt_user_j2, target) in opt_user_j.iter().zip(&mut mat_iter) {
					*target = (((T::LIKELIHOOD_PRECISION * opt_user_j1) >> T::FRACTIONAL_BITS)
						* opt_user_j2) >> T::FRACTIONAL_BITS;
				}
			}
			assert!(mat_iter.next().is_none());

			for (source, target) in precision_ii
				.iter()
				.zip(mat.iter_mut().step_by(T::EMBEDDING_DIM as usize + 1))
			{
				*target += source;
			}

			for ((((target, precision_ui1), opt_user_j1), precision_ii1), mean_i1) in vec
				.iter_mut()
				.zip(precision_ui)
				.zip(&opt_user_j)
				.zip(precision_ii)
				.zip(mean_i)
			{
				*target = ((precision_rating - precision_ui1) * opt_user_j1
					+ precision_ii1 * mean_i1)
					>> T::FRACTIONAL_BITS;
			}

			std::mem::replace(&mut opt_item_i, Self::lin_solve(mat.clone(), &vec));
		}

		(opt_item_i, opt_user_j, other_correlations)
	}

	fn lin_solve(matrix: Vec<i64>, b: &[i64]) -> Vec<i64> {
		let (lu, perm) = Self::decompose_lu(matrix);

		// Apply `perm` to b.
		let b = perm.iter().map(|i| b[*i]).collect();

		let b = Self::lu_forward_substitution(&lu, b);
		Self::back_substitution(&lu, b)
	}

	fn decompose_lu(matrix: Vec<i64>) -> (Vec<i64>, Vec<usize>) {
		let mut lu = matrix;
		let K = T::EMBEDDING_DIM as usize;
		let mut perm = (0..K).collect::<Vec<_>>();

		for index in 0..K {
			let mut curr_max_idx = index;
			let mut curr_max = lu[K * curr_max_idx + curr_max_idx];

			for i in (curr_max_idx + 1)..K {
				if lu[K * i + index].abs() > curr_max.abs() {
					curr_max = lu[K * i + index];
					curr_max_idx = i;
				}
			}
			assert!(curr_max != 0);

			Self::swap_rows(&mut lu, index, curr_max_idx);
			perm.swap(index, curr_max_idx);
			for i in (index + 1)..K {
				let mult = (lu[K * i + index] << T::FRACTIONAL_BITS) / curr_max;
				lu[K * i + index] = mult;
				for j in (index + 1)..K {
					lu[K * i + j] =
						lu[K * i + j] - ((mult * lu[K * index + j]) >> T::FRACTIONAL_BITS);
				}
			}
		}

		(lu, perm)
	}

	fn swap_rows(matrix: &mut [i64], a: usize, b: usize) {
		let K = T::EMBEDDING_DIM as usize;
		if a != b {
			unsafe {
				let row_a =
					std::slice::from_raw_parts_mut(matrix.as_mut_ptr().offset((K * a) as isize), K);
				let row_b =
					std::slice::from_raw_parts_mut(matrix.as_mut_ptr().offset((K * b) as isize), K);

				for (x, y) in row_a.into_iter().zip(row_b.into_iter()) {
					std::mem::swap(x, y);
				}
			}
		}
	}

	fn lu_forward_substitution(lu: &[i64], b: Vec<i64>) -> Vec<i64> {
		let K = T::EMBEDDING_DIM as usize;
		let mut x = b;

		for (i, row) in lu.chunks(K).enumerate().skip(1) {
			let adjustment = row
				.iter()
				.take(i)
				.cloned()
				.zip(x.iter().cloned())
				.fold(0i64, |sum, (l, x)| sum + ((l * x) >> T::FRACTIONAL_BITS));

			x[i] = x[i] - adjustment;
		}
		x
	}

	fn back_substitution(m: &[i64], y: Vec<i64>) -> Vec<i64> {
		let K = T::EMBEDDING_DIM as usize;
		let mut x = vec![0i64; K];

		unsafe {
			for i in (0..K).rev() {
				let mut holding_u_sum = 0i64;
				for j in (i + 1..K).rev() {
					holding_u_sum = holding_u_sum + *m.get_unchecked(K * i + j) * x[j];
				}

				let diag = *m.get_unchecked((K + 1) * i);
				assert!(diag != 0);
				x[i] = ((y[i] << T::FRACTIONAL_BITS) - holding_u_sum) / diag;
			}
		}

		x
	}
}
