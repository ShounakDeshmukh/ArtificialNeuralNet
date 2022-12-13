use ndarray::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
pub mod mnist_loader;

type Matrix = ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>;

fn main() {
    let ((mut x_train, mut y_train), (mut x_test, mut y_test)) = mnist_loader::load();

    x_train = x_train.reversed_axes(); //Transpose

    // print!("{:?}", x_train.shape());
    // print!("{:?}",x_train.slice(s!(..,0)).shape())
    let ((W1, B1), (W2, B2)) = init_weights();
}

fn init_weights() -> ((Matrix, Matrix), (Matrix, Matrix)) {
    let mut W1 = ndarray::Array2::random((10, 784), Uniform::new(0.0, 1.0));

    print!("{:?}", W1);

    let mut B1 = ndarray::Array2::random((10, 1), Uniform::new(0.0, 1.0));

    print!("{:?}", B1);

    let mut W2 = ndarray::Array2::random((10, 10), Uniform::new(0.0, 1.0));
    print!("{:?}", W2);

    let mut B2 = ndarray::Array2::random((10, 1), Uniform::new(0.0, 1.0));

    print!("{:?}", B2);

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
    return Z;
}

fn forward_propogation(W1: Matrix, B1: Matrix, W2: Matrix, B2: Matrix, x_train: Matrix) {
    let mut Z1 = W1.dot(&x_train) + &B1;
    let A1 = ReLU(Z1);

    
}
