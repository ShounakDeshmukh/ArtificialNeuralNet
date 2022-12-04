pub mod mnist_loader;

fn main() {
    let ((x_train, y_train), (x_test, y_test)) = mnist_loader::load();
    println!("{:?}", x_train);
}
