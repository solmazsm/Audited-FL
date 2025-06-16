# Default parameters
$n_core = 100
$aggr_method = "fedcls"  # or "fedic"
$gauss_var = 0.01
$lr = 0.01
$momentum = 0.9
$weight_decay = 1e-4
$epochs = 100
$batch_size = 32

# Parse command line arguments
for ($i = 0; $i -lt $args.Count; $i += 2) {
    switch ($args[$i]) {
        "--n_core" { $n_core = $args[$i+1] }
        "--aggr_method" { $aggr_method = $args[$i+1] }
        "--gauss_var" { $gauss_var = $args[$i+1] }
        "--lr" { $lr = $args[$i+1] }
        "--momentum" { $momentum = $args[$i+1] }
        "--weight_decay" { $weight_decay = $args[$i+1] }
        "--epochs" { $epochs = $args[$i+1] }
        "--batch_size" { $batch_size = $args[$i+1] }
        default { Write-Host "Unknown parameter: $($args[$i])"; exit 1 }
    }
}

# Run the training using mpiexec
$command = "mpiexec -n $n_core python src/gausschain_imagenet.py " + `
    "--aggr_method $aggr_method " + `
    "--gauss_var $gauss_var " + `
    "--lr $lr " + `
    "--momentum $momentum " + `
    "--weight_decay $weight_decay " + `
    "--epochs $epochs " + `
    "--batch_size $batch_size"

Write-Host "Running command: $command"
Invoke-Expression $command 