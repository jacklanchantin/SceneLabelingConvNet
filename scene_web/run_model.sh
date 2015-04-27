#!/bin/bash

URL=$1
id=$2
cd inputs; curl -O $URL; cd -

th run_model.lua inputs/`basename $URL` public/$id.jpg
