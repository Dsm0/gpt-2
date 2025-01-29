// The entry file of your WebAssembly module.

export function add(a: i32, b: i32): i32 {
  return a + b;
}

console.log("content-type: text/plain");
console.log("");
console.log("Hello, World");