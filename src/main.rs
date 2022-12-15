use ndarray::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
pub mod mnist_loader;

type Matrix = ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>;

fn main() {
    let ((x_train, mut y_train), (mut x_test, mut y_test)): (
        (mnist_loader::Mnist_array, mnist_loader::Mnist_array),
        (mnist_loader::Mnist_array, mnist_loader::Mnist_array),
    ) = mnist_loader::load();
    let mut x_train: Matrix = x_train.into_dimensionality::<Ix2>().unwrap();
    x_train = x_train.reversed_axes(); //Transpose

    print!("{:?}", x_train);
    // print!("{:?}",x_train.slice(s!(..,0)).shape())
    let ((W1, B1), (W2, B2)): ((Matrix, Matrix), (Matrix, Matrix)) = init_weights();
    forward_propogation(W1, B1, W2, B2, x_train);
}

fn init_weights() -> ((Matrix, Matrix), (Matrix, Matrix)) {
    let mut W1: Matrix = ndarray::Array2::random((10, 784), Uniform::new(-1.0, 0.5));

    print!("{:?}", W1);
    println!();

    let mut B1: Matrix = ndarray::Array2::random((10, 1), Uniform::new(-1.0, 0.5));

    // print!("{:?}", B1);

    let mut W2: Matrix = ndarray::Array2::random((10, 10), Uniform::new(-1.0, 0.5));
    // print!("{:?}", W2);

    let mut B2: Matrix = ndarray::Array2::random((10, 1), Uniform::new(-1.0, 0.5));

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

fn softmax(mut Z: Matrix) -> Matrix {
    for column_num in 0..Z.ncols() {
        let exp_Z_sum: f32 = Z.clone().column(column_num).iter().map(|x| x.exp()).sum();
        println!("{}", exp_Z_sum);

        for element in Z.column_mut(column_num) {
            *element = element.exp() / exp_Z_sum;
        }
    }

    // println!();

    println!();
    print!("{:?}", Z);

    return Z;
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
