_lab_completions() {
    local cur prev demos drop_types
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    demos="oscillator coupled pendulum double kepler cyclotron drop emwave rays"
    drop_types="cube coin rod"

    if [[ "$prev" == "drop" ]]; then
        COMPREPLY=($(compgen -W "$drop_types" -- "$cur"))
    elif [[ "${COMP_WORDS[*]}" == *main.py* && $COMP_CWORD -ge 2 ]]; then
        COMPREPLY=($(compgen -W "$demos" -- "$cur"))
    else
        COMPREPLY=()
    fi
}

complete -F _lab_completions -o default python
complete -F _lab_completions -o default python3
