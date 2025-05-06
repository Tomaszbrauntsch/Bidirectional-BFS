#!/usr/bin/env bash
set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH_DIR="$PROJECT_ROOT/graphs"
GRAPHS=(1k.bin 10k.bin 50k.bin 100k.bin)
SRC=0
NP=4

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

OUTFILE="$PROJECT_ROOT/benchmark_results.csv"
echo "version,graph,time_sec,logfile" > "$OUTFILE"

for ver in "${VERSIONS[@]}"; do
  VDIR="$PROJECT_ROOT/$ver"
  echo
  echo "========== Processing $ver =========="
  pushd "$VDIR" >/dev/null

    # build
    echo "--> make clean && make in $ver"
    make clean
    make

    EXE="./${EXE_NAME[$ver]}"
    if [[ ! -x "$EXE" ]]; then
      echo "ERROR: executable $EXE not found in $ver" >&2
      popd >/dev/null
      exit 1
    fi

    # ensure logs/ exists
    mkdir -p logs

    for g in "${GRAPHS[@]}"; do
      num=${g%k.bin}
      DST=$(( num * 1000 - 1 ))

      LOGFILE="logs/${g%.bin}.log"
      echo "--> $ver on $g (src=$SRC dst=$DST) → $LOGFILE"

      if ${USE_MPI[$ver]}; then
        cmd=( mpirun -n $NP "$EXE" "$GRAPH_DIR/$g" "$SRC" "$DST" )
      else
        cmd=( "$EXE" "$GRAPH_DIR/$g" "$SRC" "$DST" )
      fi

      # run & save everything to the per-run log
      "${cmd[@]}" &> "$LOGFILE"

      # extract time from that log
      TIME=$(awk 'BEGIN{IGNORECASE=1}
        /search time|^\[time\]/{ 
          for(i=1;i<=NF;i++) 
            if ($i ~ /^[0-9]*\.[0-9]+$/){ print $i; exit }
        }' "$LOGFILE"
      )
      TIME=${TIME:-NA}

      # append to CSV (logfile path relative to project root)
      RELLOG="${ver}/${LOGFILE}"
      echo "$ver,$g,$TIME,$RELLOG" >> "$OUTFILE"
    done

  popd >/dev/null
done

echo
echo "🏁 Done. Results in $OUTFILE"
