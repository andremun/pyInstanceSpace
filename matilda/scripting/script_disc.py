"""Display software disclaimer and citation information for a given software or file.

The `script_disc` function within this module is used to print out standardized
disclaimer and citation information related to the software, including authorship,
copyright, and how to cite the software in academic or professional contexts. This
function is designed to be used as part of software distribution, ensuring that
users are aware of the terms under which the software is provided and how to
properly acknowledge its use.
"""

def script_disc(filename: str) -> None:
    """Display software disclaimer and citation information.

    Args
        filename: Name of the file or software being described.
    """
    print("-" * 73)
    print()
    print(f" '{filename}' ")
    print()
    print(" Code by: Mario Andres Muñoz Acosta")
    print("          School of Mathematics and Statistics")
    print("          The University of Melbourne")
    print("          Australia")
    print("          2019")
    print()
    print(" Copyright: Mario A. Muñoz")
    print()
    print("-" * 73)
    print()
    print(" If using this software, please cite as: ")
    print()
    print(
        " Mario Andrés Muñoz, & Kate Smith-Miles. "
        "andremun/InstanceSpace: February 2021 Update (Version v0.2-beta). "
        "Zenodo. http://doi.org/10.5281/zenodo.4521336",
    )
    print()
    print("-" * 73)
    print()
    print(" DISCLAIMER:")
    print()
    print(" THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY")
    print(" APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT")
    print(' HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY')
    print(" OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,")
    print(" THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR")
    print(" PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM")
    print(" IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF")
    print(" ALL NECESSARY SERVICING, REPAIR OR CORRECTION.")
    print()
    print(" IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING")
    print(" WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS")
    print(
        " THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY",
    )
    print(" GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE")
    print(" USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF")
    print(" DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD")
    print(" PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),")
    print(" EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF")
    print(" SUCH DAMAGES.")
    print()
