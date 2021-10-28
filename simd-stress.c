// SPDX-License-Identifier: GPL-2.0-only
/*
 * gcc -O3 -Wall -W -static -march=skylake-avx512 simd_stress.c -o simd_stress -lpthread
 */
#define _GNU_SOURCE
#include <stdio.h>		/* printf(3) */
#include <stdlib.h>		/* random(3) */
#include <sched.h>		/* CPU_SET */
#include <immintrin.h>
#include <stdint.h>
#include <time.h>
#include <getopt.h>
#include <pthread.h>
#include <err.h>
#include <sys/syscall.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#pragma GCC target("avx512ifma")
#define WORKLOAD_NAME "IFMA512"
#define BITS_PER_VECTOR		512
#define BYTES_PER_VECTOR	(BITS_PER_VECTOR / 8)
#define WORDS_PER_VECTOR        (BITS_PER_VECTOR / 16)
#define DWORD_PER_VECTOR	(BITS_PER_VECTOR / 32)

#define ITERATIONS 256
/* AVX512: 256 ITERATIONS * 64 BYTES_PER_VECTOR * 1-byte = 16 KB */
#pragma GCC optimize ("unroll-loops")

extern void init_i32_max_tile_buffer(uint8_t *buf);

int simd_ins = 0, thread_nr = 0, duration_sec = 0;

struct thread_data {
	u_int8_t *input_x;
	int8_t *input_y;
	int32_t *input_z;
	int16_t *input_ones;
	int32_t *output;
	pthread_t tid;
};

static inline void cpuid(uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx)
{
	asm volatile("cpuid;"
		     : "=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx)
		     : "0" (*eax), "2" (*ecx));
}

static __attribute__((noinline)) unsigned long long rdtsc(void)
{
	unsigned hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

/* vpmadd52huq zmm, zmm, zmm */
static void madd52hi_epu64(struct thread_data *dp)
{
	int i;

	for (i = 0; i < ITERATIONS; ++i) {

		__m512i vx, vy, vz, voutput;

		vx = _mm512_loadu_si512((void *)(dp->input_x + i * BYTES_PER_VECTOR));
		vy = _mm512_loadu_si512((void *)(dp->input_y + i * BYTES_PER_VECTOR));
		vz = _mm512_loadu_si512((void *)(dp->input_z + i * DWORD_PER_VECTOR));

		voutput = _mm512_madd52hi_epu64(vz, vx, vy);
		_mm512_storeu_si512((void *)(dp->output + i * DWORD_PER_VECTOR), voutput);
	}
}

/* vfmadd231pd zmm, zmm, zmm */
static void fmadd_pd(struct thread_data *td)
{
	int i;

	for (i = 0; i < ITERATIONS; ++i) {

		__m512d vx, vy, vz, voutput;

		vx = _mm512_loadu_pd((void *)(td->input_x + i * BYTES_PER_VECTOR));
		vy = _mm512_loadu_pd((void *)(td->input_y + i * BYTES_PER_VECTOR));
		vz = _mm512_loadu_pd((void *)(td->input_z + i * DWORD_PER_VECTOR));

		voutput = _mm512_fmadd_pd(vz, vx, vy);
		_mm512_storeu_pd((void *)(td->output + i * DWORD_PER_VECTOR), voutput);
	}
}

int nop_per_loop = 10000000;

static void nop_loop(void)
{
	int i = 0;

	while(i++ < nop_per_loop);
}

static void run_simd(int type, struct thread_data *td)
{
	if (type == 0) {
		nop_loop();
	} else if (type == 1) {
		madd52hi_epu64(td);
	} else if (type == 2) {
		fmadd_pd(td);
	} else
		exit(1);
}

char *progname;

static void help(void)
{
	fprintf(stderr,
		"usage: %s [OPTIONS]\n"
		"%s runs avx/amx stress test\n"
		"  -d, --duration\n"
		"  -t, --thread-count\n"
		"  -l, --nop-per-loop\n"
		"  -i, --instruction-type [0:nop_loop 1:madd 2:fmadd]\n", progname, progname);
}

static char* instruction_desc[] = {
	"nop_loop",
	"vpmadd52huq",
	"vfmadd231pd",
};

char *option_string = "d:t:l:i:h";
static struct option long_options[] = {
	{"duration", required_argument, 0, 'd'},
	{"thread-count", required_argument, 0, 't'},
	{"nop-per-loop", required_argument, 0, 'l'},
	{"instruction-type", required_argument, 0, 'i'},
	{"help", no_argument, 0, 'h'},
	{0, 0, 0, 0}
};

static void parse_options(int ac, char **av)
{
	int c;

	progname = av[0];

	while (1) {
		int option_index = 0;

		c = getopt_long(ac, av, option_string,
				long_options, &option_index);
		if (c == -1)
			break;
		switch(c) {
		case 'd':
			duration_sec = atoi(optarg);
			printf("Running %d seconds...\n", duration_sec);
			break;
		case 't':
			thread_nr = atoi(optarg);
			printf("Launching %d threads...\n", thread_nr);
			break;
		case 'i':
			simd_ins = atoi(optarg);
			printf("Instruction type %d...\n", simd_ins);
			break;
		case 'l':
			nop_per_loop = atoi(optarg);
			printf("Nop per noop set to %d...\n", nop_per_loop);
			break;
		case 'h':
			help();
			break;
		default:
			break;
		}
	}
}

void *worker_thread(void *arg)
{
	int64_t start;
	unsigned long long total = 0, before, after;
	unsigned long loops = 0;
	struct thread_data *td = (struct thread_data *)arg;

	printf("Start running with %d seconds of instruction:%s\n",
		duration_sec, instruction_desc[simd_ins]);

	start = time(NULL);
	while(time(NULL) < start + 1 + duration_sec) {
		before = rdtsc();
		run_simd(simd_ins, td);
		after = rdtsc();
		total += after - before;
		loops++;
	}
	printf("Throughput %ld lps\n", (loops / duration_sec));

	return NULL;
}

int init_thread_data(struct thread_data *td)
{
	td->input_x = (u_int8_t *) calloc(ITERATIONS, BYTES_PER_VECTOR * sizeof(u_int8_t));
	td->input_y = (int8_t *) calloc(ITERATIONS, BYTES_PER_VECTOR * sizeof(int8_t));
	td->input_z = (int32_t *) calloc(ITERATIONS, DWORD_PER_VECTOR * sizeof(int32_t));
	if (!td->input_x || !td->input_y || !td->input_z)
		exit(1);

	td->output = (int32_t *) calloc(ITERATIONS, DWORD_PER_VECTOR * sizeof(int32_t));
	if (!td->output)
		exit(1);

	return 0;
}

int main(int argc, char *argv[])
{
	struct thread_data *td, *this_td;
	int i, ret;

	parse_options(argc, argv);

	td = calloc(thread_nr, sizeof(struct thread_data));
	if (!td)
		exit(1);

	for (i = 0; i < thread_nr; i++) {
		pthread_t tid;
		this_td = td + i;
		init_thread_data(this_td);
		ret = pthread_create(&tid, NULL, worker_thread,
				     this_td);
		if (ret) {
			fprintf(stderr, "error %d from pthread_create\n", ret);
			exit(1);
		}
		this_td->tid = tid;
	}

	for (i = 0; i < thread_nr; i++) {
		this_td = td + i;
		pthread_join(this_td->tid, NULL);
	}
	return 0;
}
