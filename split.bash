#!/bin/bash

## N is the desired number of testing examples
N=$1
shift

## Grab data from standard input
cat | shuf > temp.$$.txt

## All but last N
head -n -${N} temp.$$.txt > temp.$$.train.txt

## Only last N
tail -n ${N} temp.$$.txt > temp.$$.test.txt

## Run command with split data appended
$@ temp.$$.train.txt temp.$$.test.txt

## Remove temporary files
rm temp.$$.*