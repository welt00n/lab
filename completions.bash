# Bash completions for the lab physics framework.
# Source this file:  source completions.bash

_lab_main_completions() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local demos="oscillator coupled pendulum double kepler cyclotron drop emwave rays"
    local drop_types="cube coin"

    if [[ "$prev" == "drop" ]]; then
        COMPREPLY=($(compgen -W "$drop_types" -- "$cur"))
    else
        COMPREPLY=($(compgen -W "$demos" -- "$cur"))
    fi
}

_lab_drop_experiment_completions() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local flags="--nh --na --hmin --hmax --axis --workers --gpu --live"
    local axes="x y z"

    case "$prev" in
        --axis)
            COMPREPLY=($(compgen -W "$axes" -- "$cur"))
            return ;;
        --nh|--na|--hmin|--hmax|--workers)
            COMPREPLY=()
            return ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "$flags" -- "$cur"))
    fi
}

# main.py demos
complete -F _lab_main_completions -o default python main.py
complete -F _lab_main_completions -o default python3 main.py

# experiment scripts
complete -F _lab_drop_experiment_completions -o default experiments/drop_coin.py
complete -F _lab_drop_experiment_completions -o default experiments/drop_cube.py

# also match when invoked via python
_lab_experiment_dispatch() {
    local script="${COMP_WORDS[1]}"
    case "$script" in
        main.py)
            _lab_main_completions ;;
        experiments/drop_*.py|*/drop_*.py)
            _lab_drop_experiment_completions ;;
    esac
}

complete -F _lab_experiment_dispatch -o default python
complete -F _lab_experiment_dispatch -o default python3
