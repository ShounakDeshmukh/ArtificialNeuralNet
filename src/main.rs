use ndarray::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
pub mod mnist_loader;

type Matrix = ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>;

fn main() {
    let ((x_train, y_train), (x_test, y_test)): (
        (mnist_loader::Mnist_array, mnist_loader::Mnist_array),
        (mnist_loader::Mnist_array, mnist_loader::Mnist_array),
    ) = mnist_loader::load();
    let mut x_train: Matrix = x_train.into_dimensionality::<Ix2>().unwrap();
    let y_train: Matrix = y_train.into_dimensionality::<Ix2>().unwrap();

    x_train = x_train.reversed_axes(); //Transpose

    let ((mut W1, mut B1), (mut W2, mut B2)): ((Matrix, Matrix), (Matrix, Matrix));
    let ((mut Z1, mut A1), (mut Z2, mut A2)): ((Matrix, Matrix), (Matrix, Matrix));
    let ((mut delta_W1, mut delta_B1), (mut delta_W2, mut delta_B2)): (
        (Matrix, Matrix),
        (Matrix, Matrix),
    );

    ((W1, B1), (W2, B2)) = init_weights();
    for iteration in 0..2 {
        ((Z1, A1), (Z2, A2)) = forward_propogation(
            W1.clone(),
            B1.clone(),
            W2.clone(),
            B2.clone(),
            x_train.clone(),
        );

        ((delta_W1, delta_B1), (delta_W2, delta_B2)) =
            back_propogation(Z1, A1, A2, W2.clone(), x_train.clone(), y_train.clone());

        ((W1, B1), (W2, B2)) = update_weights_biases(
            delta_W1, delta_B1, delta_W2, delta_B2, W1, B1, W2, B2, 0.001,
        );

        println!("{:?}", W1);
        println!("{:?}", W2);
        println!("Iteration: {}", iteration);
    }
}

fn init_weights() -> ((Matrix, Matrix), (Matrix, Matrix)) {
    let W1: Matrix = ndarray::Array2::random((10, 784), Uniform::new(-1.0, 0.5));

    print!("{:?}", W1);
    println!();

    let B1: Matrix = ndarray::Array2::random((10, 1), Uniform::new(-1.0, 0.5));

    // print!("{:?}", B1);

    let W2: Matrix = ndarray::Array2::random((10, 10), Uniform::new(-1.0, 0.5));
    // print!("{:?}", W2);

    let B2: Matrix = ndarray::Array2::random((10, 1), Uniform::new(-1.0, 0.5));

    // print!("{:?}", B2);

    ((W1, B1), (W2, B2))
}

fn ReLU(mut Z: Matrix) -> Matrix {
    for element in Z.iter_mut() {
        if element < &mut 0.0 {
            *element = 0.0;
        } else {
            continue;
        }
    }
    print!("{:?}", Z);
    println!();
    return Z;
}

fn ReLU_Derivative(mut Z: Matrix) -> Matrix {
    for element in Z.iter_mut() {
        if element > &mut 0.0 {
            *element = 1.0;
        } else {
            continue;
        }
    }
    Z
}

fn softmax(mut Z: Matrix) -> Matrix {
    for column_num in 0..Z.ncols() {
        let exp_Z_sum: f32 = Z.clone().column(column_num).iter().map(|x| x.exp()).sum();
        // println!("{}", exp_Z_sum);

        for element in Z.column_mut(column_num) {
            *element = element.exp() / exp_Z_sum;
        }
    }

    return Z;
}
fn one_hot_encode(y_train: Matrix) -> Matrix {
    let mut one_hot_y: Matrix = Array2::zeros((y_train.shape()[0], 10));

    println!("{:?}", one_hot_y.shape());
    // (60,000;10)

    for (index, element) in y_train.indexed_iter() {
        one_hot_y[[index.0, *element as usize]] = 1.0
    }
    print!("{:?}", one_hot_y);

    one_hot_y.reversed_axes() //Return transposed matrix
}

fn forward_propogation(
    W1: Matrix,
    B1: Matrix,
    W2: Matrix,
    B2: Matrix,
    x_train: Matrix,
) -> ((Matrix, Matrix), (Matrix, Matrix)) {
    let mut Z1: Matrix = W1.dot(&x_train) + &B1;
    let A1: Matrix = ReLU(Z1.clone());
    let mut Z2: Matrix = W2.dot(&A1) + &B2;
    let A2: Matrix = softmax(Z2.clone());

    return ((Z1, A1), (Z2, A2));
}

fn back_propogation(
    Z1: Matrix,
    A1: Matrix,
    A2: Matrix,
    W2: Matrix,
    x_train: Matrix,
    y_train: Matrix,
) -> ((Matrix, Matrix), (Matrix, Matrix)) {
    let n: f32 = y_train.shape()[0] as f32;
    // n=60000
    let one_hot_y: Matrix = one_hot_encode(y_train);
    let difference_pred_actual: Matrix = A2 - one_hot_y;

    let delta_W2: Matrix = (1.0 / n) * difference_pred_actual.dot(&A1.reversed_axes());

    let delta_B2: Matrix = (1.0 / n)
        * difference_pred_actual
            .sum_axis(Axis(2))
            .into_dimensionality::<Ix2>()
            .unwrap();

    let delta_Z1: Matrix = W2.reversed_axes().dot(&difference_pred_actual) * ReLU_Derivative(Z1);
    //Element wise multiply

    let delta_W1: Matrix = (1.0 / n) * difference_pred_actual.dot(&x_train.reversed_axes());

    let delta_B1: Matrix = (1.0 / n)
        * delta_Z1
            .sum_axis(Axis(2))
            .into_dimensionality::<Ix2>()
            .unwrap();

    return ((delta_W1, delta_B1), (delta_W2, delta_B2));
}

fn update_weights_biases(
    delta_W1: Matrix,
    delta_B1: Matrix,
    delta_W2: Matrix,
    delta_B2: Matrix,
    W1: Matrix,
    B1: Matrix,
    W2: Matrix,
    B2: Matrix,
    Learning_rate: f32,
) -> ((Matrix, Matrix), (Matrix, Matrix)) {
    let W1: Matrix = W1 - Learning_rate * delta_W1;
    let B1: Matrix = B1 - Learning_rate * delta_B1;
    let W2: Matrix = W2 - Learning_rate * delta_W2;
    let B2: Matrix = B2 - Learning_rate * delta_B2;

    ((W1, B1), (W2, B2))
}

fn Training() {}
