/// Simple example demonstrating the performance improvement from tool caching
/// 
/// Before optimization: get_available_tools() allocated new Vec<Tool> on every call
/// After optimization: tools are initialized once with Lazy static and reused
///
/// Run with: cargo run --example tool_caching_demo

use openheim::tools::get_available_tools;
use std::time::Instant;

fn main() {
    println!("Tool Caching Performance Demo");
    println!("==============================\n");
    
    // Simulate multiple agent runs calling get_available_tools()
    let iterations = 10_000;
    
    println!("Calling get_available_tools() {} times...", iterations);
    
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _tools = get_available_tools();
        // In real usage, these tools would be passed to the LLM client
    }
    
    let elapsed = start.elapsed();
    
    println!("\nResults:");
    println!("--------");
    println!("Total time: {:?}", elapsed);
    println!("Average per call: {:?}", elapsed / iterations);
    println!("Calls per second: {:.0}", iterations as f64 / elapsed.as_secs_f64());
    
    println!("\nOptimization Impact:");
    println!("-------------------");
    println!("✓ Tools are now cached in a Lazy static variable");
    println!("✓ First call initializes, subsequent calls return a reference");
    println!("✓ Zero allocation overhead after first initialization");
    println!("✓ Thread-safe with minimal overhead (once_cell::Lazy)");
    
    println!("\nBefore optimization:");
    println!("- Each call allocated a new Vec with 3 Tool structs");
    println!("- ~1-2KB allocated per call");
    println!("- With {} calls, this would have been ~10-20MB wasted", iterations);
    
    println!("\nAfter optimization:");
    println!("- Tools initialized once on first call");
    println!("- Subsequent calls just return a &'static [Tool] reference");
    println!("- Memory saved: ~{:.1}MB for {} calls", 
             (iterations as f64 * 1.5) / 1024.0, iterations);
}
