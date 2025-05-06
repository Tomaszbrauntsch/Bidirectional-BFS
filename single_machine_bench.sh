#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH_DIR="$PROJECT_ROOT/graphs"
GRAPHS=(1k.bin 10k.bin 50k.bin 100k.bin)
SRC=0

# How many MPI ranks to use on this single machine:
NP=4

VERSIONS=(v1 v2 v3 v4)
declare -A EXE_NAME=(
  [v1]="bibfs_serial"
  [v2]="mpi_bibfs_bitset"
  [v3]="bibfs_cuda_only"
  [v4]="mpi_bibfs"
)
# Which versions should be launched via mpirun:
declare -A USE_MPI=(
  [v1]=false
  [v2]=true
  [v3]=false
  [v4]=true
)

OUTFILE="$PROJECT_ROOT/benchmark_results.csv"
echo "version,graph,time_sec" > "$OUTFILE"

# === LOOP ===
for ver in "${VERSIONS[@]}"; do
  VDIR="$PROJECT_ROOT/$ver"
  echo
  echo ">>> Building $ver"
  pushd "$VDIR" >/dev/null

    make clean
    make

    EXE_PATH="$VDIR/${EXE_NAME[$ver]}"
    if [[ ! -x "$EXE_PATH" ]]; then
      echo "ERROR: $EXE_PATH not found!" >&2
      exit 1
    fi
  popd >/dev/null

  for g in "${GRAPHS[@]}"; do
    # compute dst = (number * 1000) - 1
    num=${g%k.bin}       
    DST=$(( num*1000 - 1 ))

    echo "--> $ver on $g (src=$SRC dst=$DST)"
    if ${USE_MPI[$ver]}; then
      CMD=( mpirun -n $NP "$EXE_PATH" "$GRAPH_DIR/$g" "$SRC" "$DST" )
    else
      CMD=( "$EXE_PATH" "$GRAPH_DIR/$g" "$SRC" "$DST" )
    fi

    OUTPUT=$("${CMD[@]}" 2>&1)

    # Extract a decimal number after "Search time:" or "[Time]"
    TIME=$(echo "$OUTPUT" | \
      awk 'BEGIN{IGNORECASE=1}
           /search time|^\[time\]/{ 
             for(i=1;i<=NF;i++) 
               if ($i ~ /^[0-9]*\.[0-9]+$/) { print $i; exit }
           }'
    )
    TIME=${TIME:-NA}

    echo "$ver,$g,$TIME" >> "$OUTFILE"
  done
done

echo
echo "✅ All benchmarks done. See $OUTFILE"
