# Evaluation Container

This directory contains code to evaluate all languages supported by MultiPL-E (and MultiPL-E-C2C)

- We use a container to sandbox evaluation (use `--network none` as directed below);
- The container packages the build and run dependencies of all supported languages;
- The evaluation script enforces timeouts on code that runs forever;
- The evaluation script uses multicore to go faster;
- The evaluation script avoids re-running identical completions;
- The evaluation script avoids re-running completions with results that are already
  generated.

## Usage

The `build` command builds the container:

```bash
make build
```

To run the container, place all of the completion files under a directory, then use the `run` command.
Use the `--volume` option to create directory mappings.

```bash
docker run --rm --network none \
    --volume ${INPUT_DIR}:/inputs:ro \
    --volume ${OUTPUT_DIR}:/outputs:rw \
    multipl-c2c-evaluation \
        --dir $inputs \
        --output-dir $outputs
```

The `$INPUT_DIR` argument should be a directory with completions. See the `test_inputs` for an example.
The `$OUTPUT_DIR` argument is the directory where results are generated. See `Makefile` `test: build` 
for a concrete example (do not include `--testing` in your actual command).


## Specifying alternate docker instance

The docker (or podman) instance can be specified:

```bash
make DOCKER_EXEC="docker" test
```

(We use Podman by default.)

## Testing mode

A testing mode is provided to verify the correctness of the evaluation container.
In this mode, the script expects all completion files to only have two completions: 
the first one expects to pass and the second one expects to fail.
Read `src/main.py` for more detail.

To build the container and run the tests, run

```bash
make test
```

## Debug Tips

If you are running into errors building the container, or running tests, from 
connection failed to directory not found, try:
```shell
podman machine stop
podman machine start
```

After a successful build you should see an image
```shell
podman image ls
>>> REPOSITORY                            TAG         IMAGE ID      CREATED            SIZE
>>> localhost/multipl-c2c-evaluation      latest      3710*****41c  1 minutes ago     5.8 GB
```

After a successful test you should see something like
```shell
100%|██████████| 19/19 [00:59<00:00,  3.16s/it]
Execution took 60.00431990623474 seconds
```