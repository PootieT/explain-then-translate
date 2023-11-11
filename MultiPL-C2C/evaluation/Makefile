DOCKER_EXEC=podman

build: Dockerfile
	${DOCKER_EXEC} build -t multipl-c2c-evaluation .

test: build
	${DOCKER_EXEC} run --rm \
		--network none \
		--volume $(PWD)/test_translation_inputs:/inputs:ro \
		--volume $(PWD)/test_translation_outputs:/outputs:rw \
		multipl-c2c-evaluation --dir /inputs --output-dir /outputs --testing

all: build test