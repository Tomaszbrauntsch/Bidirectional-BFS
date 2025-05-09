#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH_DIR="$PROJECT_ROOT/graphs"
GRAPHS=(1k.bin 10k.bin 50k.bin 100k.bin)
SRC=0
NP=4     # number of MPI ranks to use on this single machine

VERSIONS=(v1 v2 v3 v4)
declare -A EXE_NAME=(
  [v1]="bibfs_serial"
  [v2]="second_try"
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
TABLEFILE="$PROJECT_ROOT/benchmark_table.txt"
echo "version,graph,time_sec,logfile" > "$OUTFILE"

# === RUN BENCHMARKS ===
for ver in "${VERSIONS[@]}"; do
  VDIR="$PROJECT_ROOT/$ver"
  echo
  echo "===== Building and running $ver ====="
  pushd "$VDIR" >/dev/null

    make clean && make

    EXE="./${EXE_NAME[$ver]}"
    if [[ ! -x "$EXE" ]]; then
      echo "ERROR: $EXE not found in $ver" >&2
      exit 1
    fi

    mkdir -p logs
    for g in "${GRAPHS[@]}"; do
      num=${g%k.bin}
      DST=$(( num*1000 - 1 ))
      LOGFILE="logs/${g%.bin}.log"
      echo "--> $ver on $g (src=$SRC dst=$DST) → $LOGFILE"

      if ${USE_MPI[$ver]}; then
        cmd=( mpirun -n $NP "$EXE" "$GRAPH_DIR/$g" "$SRC" "$DST" )
      else
        cmd=( "$EXE" "$GRAPH_DIR/$g" "$SRC" "$DST" )
      fi

      # run and capture everything to the log
      "${cmd[@]}" &> "$LOGFILE"

      # extract time (handles decimal & scientific)
      TIME=$(awk 'BEGIN{IGNORECASE=1}
        /search time|^\[time\]|took[[:space:]]+[0-9]|gpu bfs time/ {
          for(i=1;i<=NF;i++){
            if ($i ~ /^[0-9]+(\.[0-9]*)?([eE][+-]?[0-9]+)?$/){
              print $i; exit
            }
          }
        }' "$LOGFILE"
      )
      TIME=${TIME:-NA}

      echo "$ver,$g,$TIME,$ver/$LOGFILE" >> "$OUTFILE"
    done

  popd >/dev/null
done

# === PRINT & SAVE BOXED TABLE ===
print_boxed_table(){
  local rows=()
  while IFS=, read -r c1 c2 c3 c4; do
    rows+=("$c1|$c2|$c3|$c4")
  done < "$OUTFILE"

  # compute column widths
  local w1=0 w2=0 w3=0 w4=0
  for row in "${rows[@]}"; do
    IFS='|' read -r a b c d <<< "$row"
    (( ${#a} > w1 )) && w1=${#a}
    (( ${#b} > w2 )) && w2=${#b}
    (( ${#c} > w3 )) && w3=${#c}
    (( ${#d} > w4 )) && w4=${#d}
  done

  # build border line
  local border="+"
  for w in $((w1+2)) $((w2+2)) $((w3+2)) $((w4+2)); do
    border+=$(printf '%*s' "$w" '' | tr ' ' -)"+"
  done

  # header
  echo "$border"
  IFS='|' read -r h1 h2 h3 h4 <<< "${rows[0]}"
  printf "| %-${w1}s | %-${w2}s | %-${w3}s | %-${w4}s |\n" \
         "$h1" "$h2" "$h3" "$h4"
  echo "$border"

  # data rows
  for ((i=1; i<${#rows[@]}; i++)); do
    IFS='|' read -r a b c d <<< "${rows[i]}"
    printf "| %-${w1}s | %-${w2}s | %-${w3}s | %-${w4}s |\n" \
           "$a" "$b" "$c" "$d"
  done
  echo "$border"
}

echo
echo "🏁 Benchmark complete."
echo "CSV → $OUTFILE"
echo "Table → $TABLEFILE"
echo

# print to both stdout and save to file
print_boxed_table | tee "$TABLEFILE"
