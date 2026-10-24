# -*- coding: utf-8 -*-
"""Schema-order fix-ups for the built docx (element order inside pPr, rPr, tcPr, tblPr and m:dPr; pgMar gutter;
numbering nsid length; settings zoom). Content is not changed."""
import shutil
import zipfile
from pathlib import Path

from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
ORDER = {
    f'{{{W}}}pPr': ['pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr', 'widowControl', 'numPr',
                    'suppressLineNumbers', 'pBdr', 'shd', 'tabs', 'suppressAutoHyphens', 'kinsoku', 'wordWrap',
                    'overflowPunct', 'topLinePunct', 'autoSpaceDE', 'autoSpaceDN', 'bidi', 'adjustRightInd',
                    'snapToGrid', 'spacing', 'ind', 'contextualSpacing', 'mirrorIndents', 'suppressOverlap', 'jc',
                    'textDirection', 'textAlignment', 'textboxTightWrap', 'outlineLvl', 'divId', 'cnfStyle', 'rPr',
                    'sectPr', 'pPrChange'],
    f'{{{W}}}rPr': ['rStyle', 'rFonts', 'b', 'bCs', 'i', 'iCs', 'caps', 'smallCaps', 'strike', 'dstrike', 'outline',
                    'shadow', 'emboss', 'imprint', 'noProof', 'snapToGrid', 'vanish', 'webHidden', 'color', 'spacing',
                    'w', 'kern', 'position', 'sz', 'szCs', 'highlight', 'u', 'effect', 'bdr', 'shd', 'fitText',
                    'vertAlign', 'rtl', 'cs', 'em', 'lang', 'eastAsianLayout', 'specVanish', 'oMath'],
    f'{{{W}}}tcPr': ['cnfStyle', 'tcW', 'gridSpan', 'hMerge', 'vMerge', 'tcBorders', 'shd', 'noWrap', 'tcMar',
                     'textDirection', 'tcFitText', 'vAlign', 'hideMark', 'headers', 'cellIns', 'cellDel', 'cellMerge',
                     'tcPrChange'],
    f'{{{W}}}tblPr': ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize', 'tblStyleColBandSize',
                      'tblW', 'jc', 'tblCellSpacing', 'tblInd', 'tblBorders', 'shd', 'tblLayout', 'tblCellMar',
                      'tblLook', 'tblCaption', 'tblDescription', 'tblPrChange'],
    f'{{{M}}}dPr': ['begChr', 'sepChr', 'endChr', 'grow', 'shp', 'ctrlPr'],
    f'{{{M}}}rPr': ['lit', 'nor', 'scr', 'sty', 'brk', 'aln'],
    f'{{{W}}}style': ['name', 'aliases', 'basedOn', 'next', 'link', 'autoRedefine', 'hidden', 'uiPriority', 'semiHidden',
                      'unhideWhenUsed', 'qFormat', 'locked', 'personal', 'personalCompose', 'personalReply', 'rsid',
                      'pPr', 'rPr', 'tblPr', 'trPr', 'tcPr', 'tblStylePr'],
    f'{{{W}}}settings': ['writeProtection', 'view', 'zoom', 'removePersonalInformation', 'removeDateAndTime',
                         'doNotDisplayPageBoundaries', 'displayBackgroundShape', 'printPostScriptOverText',
                         'printFractionalCharacterWidth', 'printFormsData', 'embedTrueTypeFonts', 'embedSystemFonts',
                         'saveSubsetFonts', 'saveFormsData', 'mirrorMargins', 'alignBordersAndEdges',
                         'bordersDoNotSurroundHeader', 'bordersDoNotSurroundFooter', 'gutterAtTop', 'hideSpellingErrors',
                         'hideGrammaticalErrors', 'activeWritingStyle', 'proofState', 'formsDesign', 'attachedTemplate',
                         'linkStyles', 'stylePaneFormatFilter', 'stylePaneSortMethod', 'documentType', 'mailMerge',
                         'revisionView', 'trackRevisions', 'doNotTrackMoves', 'doNotTrackFormatting', 'documentProtection',
                         'autoFormatOverride', 'styleLockTheme', 'styleLockQFSet', 'defaultTabStop', 'autoHyphenation',
                         'consecutiveHyphenLimit', 'hyphenationZone', 'doNotHyphenateCaps', 'showEnvelope',
                         'summaryLength', 'clickAndTypeStyle', 'defaultTableStyle', 'evenAndOddHeaders',
                         'bookFoldRevPrinting', 'bookFoldPrinting', 'bookFoldPrintingSheets',
                         'drawingGridHorizontalSpacing', 'drawingGridVerticalSpacing', 'displayHorizontalDrawingGridEvery',
                         'displayVerticalDrawingGridEvery', 'doNotUseMarginsForDrawingGridOrigin',
                         'drawingGridHorizontalOrigin', 'drawingGridVerticalOrigin', 'doNotShadeFormData',
                         'noPunctuationKerning', 'characterSpacingControl', 'printTwoOnOne', 'strictFirstAndLastChars',
                         'noLineBreaksAfter', 'noLineBreaksBefore', 'savePreviewPicture', 'doNotValidateAgainstSchema',
                         'saveInvalidXml', 'ignoreMixedContent', 'alwaysShowPlaceholderText', 'doNotDemarcateInvalidXml',
                         'saveXmlDataOnly', 'useXSLTWhenSaving', 'saveThroughXslt', 'showXMLTags',
                         'alwaysMergeEmptyNamespace', 'updateFields', 'hdrShapeDefaults', 'footnotePr', 'endnotePr',
                         'compat', 'docVars', 'rsids', 'mathPr', 'attachedSchema', 'themeFontLang', 'clrSchemeMapping',
                         'doNotIncludeSubdocsInStats', 'doNotAutoCompressPictures', 'forceUpgrade', 'captions',
                         'readModeInkLockDown', 'smartTagType', 'schemaLibrary', 'shapeDefaults', 'doNotEmbedSmartTags',
                         'decimalSymbol', 'listSeparator'],
}


def _local(tag):
    return tag.split('}')[1] if '}' in tag else tag


def reorder(root):
    n = 0
    for parent_tag, order in ORDER.items():
        rank = {name: i for i, name in enumerate(order)}
        for el in root.iter(parent_tag):
            kids = [k for k in el if isinstance(k.tag, str)]
            # keep only the first of duplicate children (e.g. two cantSplit or two shd)
            seen, uniq = set(), []
            for k in kids:
                key = _local(k.tag)
                if key in seen and key not in ('ins', 'del'):
                    el.remove(k)
                    n += 1
                    continue
                seen.add(key)
                uniq.append(k)
            srt = sorted(uniq, key=lambda k: rank.get(_local(k.tag), len(order)))
            if srt != uniq:
                for k in uniq:
                    el.remove(k)
                for k in srt:
                    el.append(k)
                n += 1
    return n


def fix_docx(path):
    path = Path(path)
    tmp = path.with_suffix('.fix.docx')
    zin = zipfile.ZipFile(path)
    zout = zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED)
    stats = {}
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename in ('word/document.xml', 'word/styles.xml', 'word/numbering.xml', 'word/settings.xml',
                             'word/footer1.xml'):
            root = etree.fromstring(data)
            stats[item.filename] = reorder(root)
            if item.filename == 'word/document.xml':
                for pm in root.iter(f'{{{W}}}pgMar'):
                    if pm.get(f'{{{W}}}gutter') is None:
                        pm.set(f'{{{W}}}gutter', '0')
                    for a in ('header', 'footer'):
                        if pm.get(f'{{{W}}}{a}') is None:
                            pm.set(f'{{{W}}}{a}', '708')
            if item.filename == 'word/numbering.xml':
                for ns in root.iter(f'{{{W}}}nsid'):
                    v = ns.get(f'{{{W}}}val')
                    if v and len(v) < 8:
                        ns.set(f'{{{W}}}val', v.rjust(8, '0'))
            if item.filename == 'word/settings.xml':
                for z in list(root.iter(f'{{{W}}}zoom')):
                    z.getparent().remove(z)
            if item.filename in ('word/styles.xml', 'word/document.xml'):
                for tag in (f'{{{W}}}rPr', f'{{{W}}}pPr', f'{{{W}}}style'):
                    for el in root.iter(tag):
                        if el.text and el.text.strip():
                            el.text = None
                        for ch in el:
                            if ch.tail and ch.tail.strip():
                                ch.tail = None
            data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        zout.writestr(item, data)
    zout.close()
    shutil.move(tmp, path)
    return stats
