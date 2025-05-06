#!/usr/bin/env bash
set -euo pipefail

# Configuration
GRAPH_DIR="../graphs"            # relative when inside vN/
GRAPHS=(1k.bin 10k.bin 50k.bin 100k.bin)
SRC=0
NP=4
HOSTFILE="host_file"             # used only for MPI versions

# Which versions and what binaries they produce
VERSIONS=(v1 v2 v3 v4)
declare -A EXE_NAME=(
  [v1]="bibfs_serial"
  [v2]="mpi_bibfs_bitset"
  [v3]="bibfs_cuda_only"
  [v4]="mpi_bibfs"
)
declare -A USE_MPI=(
  [v1]=false
  [v2]=true
  [v3]=false
  [v4]=true
)

OUTFILE="benchmark_results.csv"
echo "version,graph,time_sec" > "$OUTFILE"

for ver in "${VERSIONS[@]}"; do
  echo
  echo "========== Processing $ver =========="
  pushd "$ver" >/dev/null

    # 1) build
    echo "--> make clean && make in $ver"
    make clean
    make

    EXE="./${EXE_NAME[$ver]}"
    if [[ ! -x "$EXE" ]]; then
      echo "ERROR: executable $EXE not found in $ver" >&2
      popd >/dev/null
      exit 1
    fi

    # 2) run on each graph
    for g in "${GRAPHS[@]}"; do
      # derive dst: 1k->999, 10k->9999, etc.
      num=${g%k.bin}
      DST=$(( num * 1000 - 1 ))

      echo "--> $ver on $g (src=$SRC dst=$DST)"

      if ${USE_MPI[$ver]}; then
        cmd=( mpirun -n $NP -hostfile "$HOSTFILE" "$EXE" "$GRAPH_DIR/$g" "$SRC" "$DST" )
      else
        cmd=( "$EXE" "$GRAPH_DIR/$g" "$SRC" "$DST" )
      fi

      # run & capture
      OUTPUT=$("${cmd[@]}" 2>&1)

      # extract time (looks for "Search time:" or "[Time]")
      TIME=$(echo "$OUTPUT" | \
        awk 'BEGIN{IGNORECASE=1} /search time|^\[time\]/{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+$/){print $i; exit}}'
      )
      TIME=${TIME:-NA}

      echo "$ver,$g,$TIME" >> "../$OUTFILE"
    done

  popd >/dev/null
done

echo
echo "🏁 Done. Results in $OUTFILE"
