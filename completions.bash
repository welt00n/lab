# Bash completions for the lab physics framework.
# Source this file:  source completions.bash

_lab_main_completions() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local experiments="coin cube"
    local flags="--nh --na --hmin --hmax --axis --gpu --live"
    local axes="x y z"

    case "$prev" in
        --axis)
            COMPREPLY=($(compgen -W "$axes" -- "$cur"))
            return ;;
        --nh|--na|--hmin|--hmax)
            COMPREPLY=()
            return ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "$flags" -- "$cur"))
    else
        COMPREPLY=($(compgen -W "$experiments" -- "$cur"))
    fi
}

# main.py experiment launcher
complete -F _lab_main_completions -o default python main.py
complete -F _lab_main_completions -o default python3 main.py

# dispatch for python/python3
_lab_experiment_dispatch() {
    local script="${COMP_WORDS[1]}"
    case "$script" in
        main.py)
            _lab_main_completions ;;
        experiments/drop_*.py|*/drop_*.py)
            _lab_main_completions ;;
    esac
}

complete -F _lab_experiment_dispatch -o default python
complete -F _lab_experiment_dispatch -o default python3
