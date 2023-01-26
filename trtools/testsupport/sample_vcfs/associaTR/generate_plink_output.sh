#!/bin/bash

plink2 \
	--pheno single_traits_for_plink.tab \
	--no-psam-pheno \
	--pheno-name trait_0 \
	--covar-name $(for i in $(seq 1 9) ; do echo "trait_${i}" ; done) \
	--glm omit-ref pheno-ids hide-covar \
	--vcf many_samples_biallelic.vcf.gz

mv plink2.trait_0.glm.linear single.plink2.trait_0.glm.linear


plink2 \
	--pheno combined_traits_for_plink.tab \
	--no-psam-pheno \
	--pheno-name trait_0 \
	--covar-name $(for i in $(seq 1 14) ; do echo "trait_${i}" ; done) \
	--glm omit-ref pheno-ids hide-covar \
	--vcf many_samples_biallelic.vcf.gz

mv plink2.trait_0.glm.linear combined.plink2.trait_0.glm.linear

