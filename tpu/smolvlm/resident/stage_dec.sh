#!/bin/bash
# Copy a compiled decoder build to the board as a duosight model folder.
# Usage: bash stage_dec.sh OUT BUILD BOARD_DIR   e.g. stage_dec.sh out_t16 build_t16 /root/fast/dec
#   OUT: models under D:/duosight-eval/decoder (lNN_mix2.cvimodel, head_mix2.cvimodel)
#   BUILD: the export with prefix-imstart.kv; embed.f16 links to the one in /root/vlm/m
#   KNOWN: known_hosts file for the board
OUT=/d/duosight-eval/decoder/$1
BUILD=/d/duosight-eval/decoder/$2
DST=$3
BOARD=root@192.168.100.108
SSHO="-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o UserKnownHostsFile=$KNOWN"
n=$(ls $OUT/*.cvimodel | wc -l)
[ "$n" = 31 ] || { echo "$OUT has $n models, not 31"; exit 1; }
ssh $SSHO $BOARD "rm -rf $DST.new && mkdir -p $DST.new && ln -s /root/vlm/m/embed.f16 $DST.new/embed.f16" || exit 1
for f in $OUT/*.cvimodel; do
	b=$(basename $f .cvimodel)
	scp -q $SSHO $f $BOARD:$DST.new/${b%_*}.cvimodel || exit 1
done
scp -q $SSHO $BUILD/prefix-imstart.kv $BOARD:$DST.new/ || exit 1
ssh $SSHO $BOARD "rm -rf $DST && mv $DST.new $DST && ls $DST | wc -l && du -shL $DST/l00.cvimodel $DST/head.cvimodel && df -h / | tail -1"
