#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH_DIR="$PROJECT_ROOT/graphs"
GRAPHS=(1k.bin 10k.bin 50k.bin 100k.bin)
SRC=0
NP=4        # number of MPI ranks

VERSIONS=(v1 v2 v3 v4)
# Executable names (adjust if different)
declare -A EXE_NAME=(
  [v1]="bibfs_serial"
  [v2]="mpi_bibfs_bitset"
  [v3]="bibfs_cuda_only"
  [v4]="mpi_bibfs"
)
# Which ones run under mpirun
declare -A USE_MPI=(
  [v1]=false
  [v2]=true
  [v3]=false
  [v4]=true
)

# Output CSV
OUTFILE="$PROJECT_ROOT/benchmark_results.csv"
echo "version,graph,time_sec,logfile" > "$OUTFILE"

# === BENCHMARK LOOP ===
for ver in "${VERSIONS[@]}"; do
  VDIR="$PROJECT_ROOT/$ver"
  echo
  echo "=== Building $ver in $VDIR ==="
  pushd "$VDIR" >/dev/null

    make clean
    make

    EXE="$VDIR/${EXE_NAME[$ver]}"
    if [[ ! -x "$EXE" ]]; then
      echo "ERROR: $EXE not found or not executable!" >&2
      popd >/dev/null
      exit 1
    fi

    # prepare logs dir
    mkdir -p logs

    for g in "${GRAPHS[@]}"; do
      num=${g%k.bin}           # "1","10","50","100"
      DST=$(( num*1000 - 1 ))  # 1k->999,10k->9999...
      LOGFILE="logs/${ver}_${g%.bin}.log"
      echo "--> $ver on $g (src=$SRC dst=$DST) → $LOGFILE"

      # pick command
      if ${USE_MPI[$ver]}; then
        CMD=( mpirun -n $NP "$EXE" "$GRAPH_DIR/$g" "$SRC" "$DST" )
      else
        CMD=( "$EXE" "$GRAPH_DIR/$g" "$SRC" "$DST" )
      fi

      # run & tee to logfile
      # shellcheck disable=SC2024
      "${CMD[@]}" 2>&1 | tee "$LOGFILE"

      # extract time from the log
      TIME=$(awk 'BEGIN{IGNORECASE=1}
                  /search time|^\[time\]/{ 
                    for(i=1;i<=NF;i++)
                      if ($i ~ /^[0-9]*\.[0-9]+$/){
                        print $i; exit
                      }
                  }' "$LOGFILE")
      TIME=${TIME:-NA}

      # record in CSV: version,graph,time,logfile (relative path)
      echo "$ver,$g,$TIME,$ver/$LOGFILE" >> "$OUTFILE"

    done

  popd >/dev/null
done

echo
echo "🏁 Done! Results: $OUTFILE"
