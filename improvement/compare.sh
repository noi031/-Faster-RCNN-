# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/xzb/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	    eval "$__conda_setup"
    else
	        if [ -f "/home/xzb/anaconda3/etc/profile.d/conda.sh" ]; then
			        . "/home/xzb/anaconda3/etc/profile.d/conda.sh"
				    else
					            export PATH="/home/xzb/anaconda3/bin:$PATH"
						        fi
fi
unset __conda_setup
# <<< conda initialize <<<
source activate py37
python compare.py -result_path r.json -target_dir ../../data/labels -image_dir ../../data/dev/images
