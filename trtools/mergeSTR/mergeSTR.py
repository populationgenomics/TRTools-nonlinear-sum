"""
Tool for merging TR VCF files generated by
the same TR genotyper.
"""

import argparse
import os
import sys
from typing import List

import cyvcf2
import numpy as np

import trtools.utils.common as common
import trtools.utils.mergeutils as mergeutils
import trtools.utils.tr_harmonizer as trh
import trtools.utils.utils as utils
from trtools import __version__

from typing import Tuple, Union, Any, Optional, List, TextIO

NOCALLSTRING = "."

# Tool-specific fields to merge. (FIELDNAME, req'd). req'd is True if merge should
# fail when all records don't have identical values for that field
INFOFIELDS = {
    trh.VcfTypes.gangstr: [("END", True), ("RU", True), ("PERIOD", True), ("REF", True), \
                           ("EXPTHRESH", True), ("STUTTERUP", False), \
                           ("STUTTERDOWN", False), ("STUTTERP", False)],
    trh.VcfTypes.hipstr: [("INFRAME_PGEOM", False), ("INFRAME_UP", False), ("INFRAME_DOWN", False), \
                          ("OUTFRAME_PGEOM", False), ("OUTFRAME_UP", False), ("OUTFRAME_DOWN", False), \
                          ("BPDIFFS", False), ("START", True), ("END", True), ("PERIOD", True), \
                          ("AN", False), ("REFAC", False), ("AC", False), ("NSKIP", False), \
                          ("NFILT", False), ("DP", False), ("DSNP", False), ("DSTUTTER", False), \
                          ("DFLANKINDEL", False)],
    trh.VcfTypes.eh: [("END", True), ("REF", True), ("REPID", True), ("RL", True), \
                      ("RU", True), ("SVTYPE", False), ("VARID", True)],
    trh.VcfTypes.popstr: [("Motif", True)],  # TODO ("RefLen", True) omitted. since it is marked as "A" incorrectly
    trh.VcfTypes.advntr: [("END", True), ("VID", True), ("RU", True), ("RC", True)]
}

# Tool-specific format fields to merge
# Not all fields currently handled
# If not listed here, it is ignored
FORMATFIELDS = {
    trh.VcfTypes.gangstr: ["DP", "Q", "REPCN", "REPCI", "RC", "ENCLREADS", "FLNKREADS", "ML", "INS", "STDERR", "QEXP"],
    trh.VcfTypes.hipstr: ["GB", "Q", "PQ", "DP", "DSNP", "PSNP", "PDP", "GLDIFF", "DSTUTTER", "DFLANKINDEL", "AB", "FS",
                          "DAB", "ALLREADS", "MALLREADS"],
    trh.VcfTypes.eh: ["ADFL", "ADIR", "ADSP", "LC", "REPCI", "REPCN", "SO"],
    trh.VcfTypes.popstr: ["AD", "DP", "PL"],
    trh.VcfTypes.advntr: ["DP", "SR", "FR", "ML"]
}


def WriteMergedHeader(vcfw: TextIO, args: Any, readers: List[cyvcf2.VCF], cmd: str, vcftype: Union[str, trh.VcfTypes]) \
        -> Union[Tuple[List[Tuple[str, bool]], List[str]], Tuple[None, None]]:
    r"""Write merged header for VCFs in args.vcfs

    Also do some checks on the VCFs to make sure merging
    is appropriate.
    Return info and format fields to use

    Parameters
    ----------
    vcfw : file object
       Writer to write the merged VCF
    args : argparse namespace
       Contains user options
    readers : list of vcf.Reader
       List of readers to merge
    cmd : str
       Command used to call this program
    vcftype : str
       Type of VCF files being merged

    Returns
    -------
    useinfo : list of (str, bool)
       List of (info field, required) to use downstream
    useformat: list of str
       List of format field strings to use downstream
    """

    def get_header_lines(field: str, reader: cyvcf2.VCF) -> List[str]:
        compare_len = 3 + len(field)
        compare_start = '##' + field.lower() + "="
        return [line for line in reader.raw_header.split('\n') if \
                line[:compare_len].lower() == compare_start]

    # Check contigs the same for all readers
    contigs = get_header_lines('contig', readers[0])
    for i in range(1, len(readers)):
        if set(get_header_lines('contig', readers[i])) != set(contigs):
            raise ValueError(
                "Different contigs found across VCF files. Make sure all "
                "files used the same reference. Consider using this "
                "command:\n\t"
                "bcftools reheader -f ref.fa.fai file.vcf.gz -o file_rh.vcf.gz")
    # Write VCF format, commands, and contigs
    vcfw.write("##fileformat=VCFv4.1\n")

    # Update commands
    for r in readers:
        for line in get_header_lines('command', r):
            vcfw.write(line + '\n')
    vcfw.write("##command=" + cmd + "\n")

    # Update sources
    sources = set.union(*[set(get_header_lines('source', reader)) for reader in readers])
    for src in sources:
        vcfw.write(src + "\n")

    for contig in contigs:
        vcfw.write(contig + "\n")

    # Write ALT fields if present
    alts = set.union(*[set(get_header_lines('alt', reader)) for reader in readers])
    for alt in alts:
        vcfw.write(alt + '\n')

    # Write INFO fields, different for each tool
    useinfo: List[Tuple[str, bool]] = []
    infos = get_header_lines('info', readers[0])
    for (field, reqd) in INFOFIELDS[vcftype]:
        this_info = [line for line in infos if 'ID=' + field + ',' in line]
        if len(this_info) == 0:
            common.WARNING("Expected info field %s not found. Skipping" % field)
        elif len(this_info) >= 2:
            common.WARNING("Found two header lines matching the info field %s. Skipping" % field)
        else:
            vcfw.write(this_info[0] + '\n')
            useinfo.append((field, reqd))

    # Write GT header
    vcfw.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
    # Write FORMAT fields, different for each tool
    useformat: List[str] = []
    formats = get_header_lines('format', readers[0])
    for field in FORMATFIELDS[vcftype]:
        this_format = [line for line in formats if 'ID=' + field + ',' in line]
        if len(this_format) == 0:
            common.WARNING("Expected format field %s not found. Skipping" % field)
        elif len(this_format) >= 2:
            common.WARNING("Found two header lines matching the format field %s. Skipping" % field)
        else:
            vcfw.write(this_format[0] + '\n')
            useformat.append(field)

    # Write sample list
    try:
        if not args.update_sample_from_file:
            samples = mergeutils.GetSamples(readers)
        else:
            filenames = [fname.split('/')[-1] for fname in args.vcfs.split(',')]
            samples = mergeutils.GetSamples(readers, filenames)
    except ValueError as ve:
        common.WARNING("Error: " + str(ve))
        return None, None
    if len(samples) == 0:
        return None, None
    header_fields = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    vcfw.write("#" + "\t".join(header_fields + samples) + "\n")
    return useinfo, useformat



def GetRefAllele(current_records: List[trh.TRRecord], mergelist: List[bool], vcfType: trh.VcfTypes) -> Optional[str]:
    r"""Get reference allele for a set of records

    Parameters
    ----------
    current_records : list of vcf.Record
       List of records being merged
    mergelist : list of bool
       Indicates whether each record is included in merge

    Returns
    -------
    ref : str
       Reference allele string. Set to None if conflicting references are found.
    """

    def DefaultKey(record: trh.TRRecord):
        return record.vcfrecord.REF

    def HipstrKey(record: trh.TRRecord):
        return record.ref_allele

    refs: List[str] = []
    ref_picker = DefaultKey
    if vcfType == trh.VcfTypes.hipstr:
        ref_picker = HipstrKey

    chrom = ""
    pos = -1
    for i in range(len(mergelist)):
        if mergelist[i]:
            chrom = current_records[i].chrom
            pos = current_records[i].pos
            refs.append(ref_picker(current_records[i]).upper())
    if len(set(refs)) != 1:
        return None
    return refs[0]




def GetAltsByKey(current_records: List[trh.TRRecord], mergelist: List[bool], key: Any):
    result = set()
    for i in range(len(mergelist)):
        if mergelist[i]:
            ralts = key(current_records[i])
            for item in ralts:
                result.add(item.upper())
    return result


def GetAltAlleles(current_records: List[trh.TRRecord], mergelist: List[bool], vcftype: Union[str, trh.VcfTypes]) \
        -> Tuple[List[str], List[np.ndarray]]:
    r"""Get list of alt alleles
    
    Parameters
    ----------
    current_records : list of vcf.Record
       List of records being merged
    mergelist : list of bool
       Indicates whether each record is included in merge
    vcftype :
        The type of the VCFs these records came from

    Returns
    -------
    (alts, mappings) : (list of str, list of np.ndarray)
       alts is a list of alternate allele strings in all uppercase.
       mappings is a list of length equal to the number of
       records being merged. For record n, mappings[n] is a numpy
       array where an allele with index i in the original
       record has an index of mappings[n][i] in the output merged record.
       (the indicies stored in the arrays are number strings for fast
       output, e.g. '1' or '2')
       For example if the output record has ref allele 'A'
       and alternate alleles 'AA,AAA,AAAA'
       and input record n has ref allele 'A' and alternate alleles
       'AAAA,AAA' then mappings[n] would be np.array(['0', '3', '2']). ::

         original index      new index
         rec_n.alleles[0] == out_rec.alleles[0] == 'A'
         rec_n.alleles[1] == out_rec.alleles[3] == 'AAAA'
         rec_n.alleles[2] == out_rec.alleles[2] == 'AAA'
    """

    def DefaultKey(record: trh.TRRecord):
        return record.vcfrecord.ALT

    def HipstrKey(record: trh.TRRecord):
        return record.alt_alleles

    alt_picker = DefaultKey
    if vcftype == trh.VcfTypes.hipstr:
        alt_picker = HipstrKey

    alts = GetAltsByKey(current_records, mergelist, alt_picker)

    if vcftype == trh.VcfTypes.eh:
        # EH alleles look like <STR42> where 42 is the
        # number of repeat units so sort accordingly
        out_alts = sorted(alts, key=lambda x: int(x[4:-1]))
    elif vcftype == trh.VcfTypes.popstr:
        # popsr alleles look like <4.2> where 4.2 is the
        # number of repeat units so sort accordingly
        out_alts = sorted(alts, key=lambda x: float(x[1:-1]))
    else:
        out_alts = sorted(alts, key=lambda x: (len(x), x))

    mappings = []
    for i in range(len(mergelist)):
        if mergelist[i]:
            ralts = alt_picker(current_records[i])
            mappings.append(
                np.array([0] + [out_alts.index(ralt.upper()) + 1 for ralt in ralts]).astype(str)
            )
    return out_alts, mappings


def GetID(idval: str) -> str:
    r"""Get the ID for a a record

    If not set, output "."

    Parameters
    ----------
    idval : str
       ID of the record

    Returns
    -------
    idval : str
       Return ID. if None, return "."
    """
    if idval is None:
        return "."
    else:
        return idval


def GetInfoItem(current_records: List[trh.TRRecord], mergelist: List[bool], info_field: str, fail: bool = True) \
        -> Optional[str]:
    """Get INFO item for a group of records

    Make sure it's the same across merged records
    if fail=True, die if items not the same.
    if fail=False, only do something if we have a rule on how to handle that field

    Parameters
    ----------
    current_records : list of vcf.Record
       List of records being merged
    mergelist : list of bool
       List of indicators of whether to merge each record
    info_field : str
       INFO field being merged
    fail : bool
       If True, throw error if fields don't have same value

    Returns
    -------
    infostring : str
       INFO string to add (key=value)
    """
    if not fail: return None  # TODO in future implement smart merging of select fields
    vals = set()
    a_merged_rec = None
    for i in range(len(mergelist)):
        if mergelist[i]:
            a_merged_rec = current_records[i]
            if info_field in dict(current_records[i].info):
                vals.add(current_records[i].info[info_field])
            else:
                raise ValueError("Missing info field %s" % info_field)
    if len(vals) == 1:
        return "%s=%s" % (info_field, vals.pop())
    else:
        common.WARNING("Incompatible values %s for info field %s at position "
                       "%s:%i" % (vals, info_field, a_merged_rec.chrom,
                                  a_merged_rec.pos))
        return None


def WriteSampleData(vcfw: TextIO, record: cyvcf2.Variant, alleles: List[str], formats: List[str],
                    format_type: List[str], mapping: np.ndarray) -> None:
    r"""Output sample FORMAT data

    Writes a string representation of the GT and other format
    fields for each sample in the record, with tabs
    in between records

    Parameters
    ----------
    vcfw : file
        File to write output to
    record : cyvcf2.Varaint
       VCF record being summarized
    alleles : list of str
       List of REF + ALT alleles
    formats : list of str
       List of VCF FORMAT items
    format_type: list of String
        The type of each format field
    mapping: np.ndarray
        See GetAltAlleles
    """
    assert "GT" not in formats  # since we will add that

    genotypes = record.genotype.array()
    not_called_samples = np.all(
        np.logical_or(genotypes[:, :-1] == -1, genotypes[:, :-1] == -2),
        axis=1
    )
    phase_chars = np.array(['/', '|'])[genotypes[:, -1]]

    # pre retrieve all the numpy arrays
    # in case that speeds up performance
    format_arrays = {}
    for format_idx, fmt in enumerate(formats):
        if format_type[format_idx] == 'String':
            format_arrays[fmt] = record.format(fmt)
        elif format_type[format_idx] == 'Float':
            format_arr = record.format(fmt)
            nans = np.isnan(format_arr)
            format_arr = format_arr.astype(str)
            format_arr[nans] = '.'
            format_arrays[fmt] = format_arr
        else:
            format_arrays[fmt] = record.format(fmt).astype(str)

    for sample_idx in range(genotypes.shape[0]):
        vcfw.write('\t')

        if not_called_samples[sample_idx]:
            vcfw.write(".")
            continue

        # Add GT
        vcfw.write(phase_chars[sample_idx].join(
            mapping[genotypes[sample_idx, :-1]]
        ))

        # Add rest of formats
        for fmt_idx, fmt in enumerate(formats):
            vcfw.write(':')
            if format_type[fmt_idx] == 'String':
                vcfw.write(format_arrays[fmt][sample_idx])
                continue
            else:
                vcfw.write(','.join(
                    format_arrays[fmt][sample_idx, :]
                ))



def MergeRecords(readers: cyvcf2.VCF, vcftype: Union[str, trh.VcfTypes], num_samples: List[int],
                 current_records: List[trh.TRRecord],
                 mergelist: List[bool], vcfw: TextIO, useinfo: List[Tuple[str, bool]],
                 useformat: List[str], format_type: List[str]) -> None:
    r"""Merge records from different files

    Merge all records with indicator set to True in mergelist
    Output merged record to vcfw

    Parameters
    ----------
    readers : list of vcf.Reader
       List of readers being merged
    vcftype :
       Type of the readers
    num_samples : list of int
       Number of samples per vcf
    current_records : list of vcf.Record
       List of current records for each reader
    mergelist : list of bool
       Indicates whether to include each reader in merge
    vcfw : file
       File to write output to
    useinfo : list of (str, bool)
       List of (info field, required) to use downstream
    useformat: list of str
       List of format field strings to use downstream
    format_type: list of String
        The type of each format field
    """
    use_ind = [i for i in range(len(mergelist)) if mergelist[i]]
    if len(use_ind) == 0: return

    chrom = current_records[use_ind[0]].chrom
    pos = str(current_records[use_ind[0]].pos)

    ref_allele = GetRefAllele(current_records, mergelist, vcftype)
    if ref_allele is None:
        common.WARNING("Conflicting refs found at {}:{}. Skipping.".format(chrom, pos))
        return

    alt_alleles, mappings = GetAltAlleles(current_records, mergelist, vcftype)

    # Set common fields
    vcfw.write(chrom)  # CHROM
    vcfw.write('\t')
    vcfw.write(pos)  # POS
    vcfw.write('\t')
    # TODO complain if records have different IDs
    vcfw.write(GetID(current_records[use_ind[0]].vcfrecord.ID))  # ID
    vcfw.write('\t')
    vcfw.write(ref_allele)  # REF
    vcfw.write('\t')
    # ALT
    if len(alt_alleles) > 0:
        vcfw.write(",".join(alt_alleles))
        vcfw.write('\t')
    else:
        vcfw.write('.\t')
    # fields which are always set to empty
    vcfw.write(".\t")  # QUAL
    vcfw.write(".\t")  # FITLER

    # INFO
    first = True
    for (field, reqd) in useinfo:
        inf = GetInfoItem(current_records, mergelist, field, fail=reqd)
        if inf is not None:
            if not first:
                vcfw.write(';')
            first = False
            vcfw.write(inf)
    vcfw.write('\t')

    # FORMAT - add GT to front
    vcfw.write(":".join(["GT"] + useformat))

    # Samples
    alleles = [ref_allele] + alt_alleles
    map_iter = iter(mappings)
    for i in range(len(mergelist)):
        if mergelist[i]:
            WriteSampleData(vcfw, current_records[i].vcfrecord, alleles, useformat,
                            format_type, next(map_iter))
        else:  # NOCALL
            if num_samples[i] > 0:
                vcfw.write('\t')
                vcfw.write('\t'.join([NOCALLSTRING] * num_samples[i]))

    vcfw.write('\n')


def getargs() -> Any:  # pragma: no cover
    parser = argparse.ArgumentParser(
        __doc__,
        formatter_class=utils.ArgumentDefaultsHelpFormatter
    )
    ### Required arguments ###
    req_group = parser.add_argument_group("Required arguments")
    req_group.add_argument("--vcfs",
                           help="Comma-separated list of VCF files to merge (must be sorted, bgzipped and indexed)",
                           type=str, required=True)
    req_group.add_argument("--out", help="Prefix to name output files", type=str, required=True)
    req_group.add_argument("--vcftype", help="Options=%s" % [str(item) for item in trh.VcfTypes.__members__], type=str,
                           default="auto")
    ### Special merge options ###
    spec_group = parser.add_argument_group("Special merge options")
    spec_group.add_argument("--update-sample-from-file",
                            help="Use file names, rather than sample header names, when merging", action="store_true")
    ### Optional arguments ###
    opt_group = parser.add_argument_group("Optional arguments")
    opt_group.add_argument("--verbose", help="Print out extra info", action="store_true")
    opt_group.add_argument("--quiet", help="Don't print out anything", action="store_true")
    ## Version argument ##
    ver_group = parser.add_argument_group("Version")
    ver_group.add_argument("--version", action="version", version='{version}'.format(version=__version__))
    ### Parse args ###
    args = parser.parse_args()
    return args


def HarmonizeIfNotNone(records: List[Optional[trh.TRRecord]], vcf_type: trh.VcfTypes):
    result = []
    for record in records:
        if record is not None:
            result.append(trh.HarmonizeRecord(vcf_type, record))
        else:
            result.append(None)

    return result


def main(args: Any) -> int:
    if not os.path.exists(os.path.dirname(os.path.abspath(args.out))):
        common.WARNING("Error: The directory which contains the output location {} does"
                       " not exist".format(args.out))
        return 1

    if os.path.isdir(args.out) and args.out.endswith(os.sep):
        common.WARNING("Error: The output location {} is a "
                       "directory".format(args.out))
        return 1

    filenames = args.vcfs.split(",")
    ### Check and Load VCF files ###
    vcfreaders = utils.LoadReaders(filenames, checkgz=True)
    if vcfreaders is None:
        return 1
    if len(vcfreaders) == 0: return 1

    num_samples = [len(reader.samples) for reader in vcfreaders]

    # WriteMergedHeader will confirm that the list of contigs is the same for
    # each vcf, so just pulling it from one here is fine
    chroms = utils.GetContigs(vcfreaders[0])

    ### Check inferred type of each is the same
    try:
        vcftype = mergeutils.GetAndCheckVCFType(vcfreaders, args.vcftype)
    except ValueError as ve:
        common.WARNING('Error: ' + str(ve))
        return 1

    ### Set up VCF writer ###
    vcfw = open(args.out + ".vcf", "w")

    useinfo, useformat = WriteMergedHeader(vcfw, args, vcfreaders, " ".join(sys.argv), vcftype)

    if useinfo is None or useformat is None:
        common.WARNING("Error writing merged header. Quitting")
        return 1

    # need to know format types to know how to convert them to strings
    format_type = []
    for fmt in useformat:
        format_type.append(vcfreaders[0].get_header_type(fmt)['Type'])

    ### Walk through sorted readers, merging records as we go ###
    current_records = mergeutils.InitReaders(vcfreaders)
    # Check if contig ID is set in VCF header for all records
    done = mergeutils.DoneReading(current_records)

    while not done:
        for vcf_num, (r, reader) in enumerate(zip(current_records, vcfreaders)):
            if r is None: continue
            if not r.CHROM in chroms:
                common.WARNING((
                                   "Error: found a record in file {} with "
                                   "chromosome '{}' which was not found in the contig list "
                                   "({})").format(filenames[vcf_num], r.CHROM, ", ".join(chroms)))
                common.WARNING("VCF files must contain a ##contig header line for each chromosome.")
                common.WARNING(
                    "If this is only a technical issue and all the vcf "
                    "files were truly built against against the "
                    "same reference, use bcftools "
                    "(https://github.com/samtools/bcftools) to fix the contigs"
                    ", e.g.: bcftools reheader -f hg19.fa.fai -o myvcf-readher.vcf.gz myvcf.vcf.gz")
                return 1
        harmonized_records = HarmonizeIfNotNone(current_records, vcftype)
        is_min = mergeutils.GetMinHarmonizedRecords(harmonized_records, chroms)
        if args.verbose: mergeutils.DebugPrintRecordLocations(current_records, is_min)
        if mergeutils.CheckMin(is_min): return 1
        MergeRecords(vcfreaders, vcftype, num_samples, harmonized_records, is_min, vcfw, useinfo,
                     useformat, format_type)
        current_records = mergeutils.GetNextRecords(vcfreaders, current_records, is_min)
        done = mergeutils.DoneReading(current_records)
    return 0


def run() -> None:  # pragma: no cover
    args = getargs()
    retcode = main(args)
    sys.exit(retcode)


if __name__ == "__main__":  # pragma: no cover
    run()
