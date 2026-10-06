"""
Sample PDF generator for 'Ask My Notes'.
Creates a multi-page college notes & syllabus PDF ('CS101_Data_Structures_Notes.pdf')
with distinct page topics to verify accurate page citations and grounded retrieval.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


def generate_sample_pdf(output_path: str = "CS101_Data_Structures_Notes.pdf") -> str:
    """
    Generates a 4-page realistic college notes & syllabus PDF.
    """
    dest = Path(output_path)
    doc = SimpleDocTemplate(
        str(dest),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12,
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8,
    )

    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=body_style,
        leftIndent=15,
        spaceAfter=4,
    )

    story = []

    # =========================================================================
    # PAGE 1: Course Overview & Complexity Theory
    # =========================================================================
    story.append(Paragraph("CS101: Data Structures and Algorithms", title_style))
    story.append(Paragraph("Unit 1: Introduction to Algorithm Complexity & Big-O", h2_style))
    story.append(
        Paragraph(
            "Computational complexity theory classifies algorithms according to how much time or memory "
            "they consume as input size N grows toward infinity. We focus on two primary resources: "
            "Time Complexity (the number of elementary operations performed) and Space Complexity "
            "(the peak auxiliary memory allocated during execution).",
            body_style,
        )
    )
    story.append(Paragraph("Asymptotic Notations & Order of Growth:", h2_style))
    story.append(
        Paragraph(
            "• <b>Big-O (O(N))</b>: Represents an asymptotic upper bound. For example, linear search is O(N).<br/>"
            "• <b>Big-Omega (Ω(N))</b>: Represents an asymptotic lower bound of the algorithm's runtime.<br/>"
            "• <b>Big-Theta (Θ(N))</b>: Represents a tight asymptotic bound where upper and lower bounds coincide.<br/>"
            "• <b>O(1) Constant Time</b>: Hash table lookups on average, array indexing, and basic arithmetic.<br/>"
            "• <b>O(log N) Logarithmic Time</b>: Binary search over a sorted array or balanced binary search tree operations.<br/>"
            "• <b>O(N log N) Linearithmic Time</b>: Optimal comparison-based sorting algorithms like Merge Sort and Heap Sort.<br/>"
            "• <b>O(N²) Quadratic Time</b>: Brute-force nested loops, Bubble Sort, Insertion Sort worst case.<br/>"
            "• <b>O(2^N) Exponential Time</b>: Recursive Fibonacci without memoization, generating all subsets of a set.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Amortized Analysis:</b> Dynamic arrays (e.g., Python lists or C++ std::vector) double their capacity "
            "whenever they fill up. While a resize operation takes O(N) copying time, it occurs infrequently enough "
            "that the average amortized cost per append operation remains strictly <b>O(1)</b>.",
            body_style,
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Linear Data Structures
    # =========================================================================
    story.append(Paragraph("Unit 2: Linear Data Structures & Memory Layouts", title_style))
    story.append(
        Paragraph(
            "Linear data structures arrange elements in sequential order. Choosing between arrays and "
            "linked representations depends fundamentally on memory layout and access patterns.",
            body_style,
        )
    )
    story.append(Paragraph("Arrays vs. Linked Lists Comparison:", h2_style))

    table_data = [
        ["Operation", "Static Array", "Singly Linked List", "Doubly Linked List"],
        ["Random Access (index i)", "O(1)", "O(N)", "O(N)"],
        ["Insert / Delete at Head", "O(N)", "O(1)", "O(1)"],
        ["Insert / Delete at Tail", "O(1) amortized", "O(N) (or O(1) with tail)", "O(1) with tail pointer"],
        ["Cache Locality", "Excellent (Contiguous)", "Poor (Scattered heap)", "Poor (Scattered heap)"],
        ["Per-element Memory Overhead", "Zero extra pointers", "1 pointer (8 bytes on x64)", "2 pointers (16 bytes on x64)"],
    ]
    t = Table(table_data, colWidths=[130, 110, 120, 140])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FAFC")),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Stacks and Queues:", h2_style))
    story.append(
        Paragraph(
            "• <b>Stack (LIFO - Last In, First Out)</b>: Fundamental operations are <code>push()</code> and <code>pop()</code>, "
            "both operating in O(1) time. Applications include the call stack execution frame in programming languages, "
            "undo mechanisms in text editors, and syntax parenthesis matching algorithms.<br/>"
            "• <b>Queue (FIFO - First In, First Out)</b>: Operations include <code>enqueue()</code> and <code>dequeue()</code>. "
            "Implemented using circular ring buffers or linked lists. Applications include operating system CPU task scheduling, "
            "print spoolers, and Breadth-First Search (BFS) frontier queues.",
            body_style,
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Hierarchical Structures & Graph Algorithms
    # =========================================================================
    story.append(Paragraph("Unit 3: Trees, Balanced Trees, and Graph Traversal", title_style))
    story.append(
        Paragraph(
            "Hierarchical non-linear data structures allow efficient searching, sorting, and relationship modeling.",
            body_style,
        )
    )
    story.append(Paragraph("Binary Search Trees (BST) & Self-Balancing Trees:", h2_style))
    story.append(
        Paragraph(
            "In a standard BST, for any node N, all keys in N's left subtree are strictly less than N.key, "
            "and all keys in the right subtree are strictly greater. In-order traversal visits nodes in ascending sorted order.<br/>"
            "• <b>Worst Case Degeneracy</b>: If elements are inserted in already-sorted order, a standard BST degenerates "
            "into a singly linked list with <b>O(N) search time</b>.<br/>"
            "• <b>AVL Trees</b>: An AVL tree enforces a strict balance factor for every node: "
            "<code>Height(LeftSubtree) - Height(RightSubtree) ∈ {-1, 0, 1}</code>. "
            "Whenever an insertion or deletion violates this invariant, tree balance is restored via O(1) tree rotations "
            "(Left rotation, Right rotation, Left-Right rotation, Right-Left rotation). "
            "Consequently, search, insertion, and deletion strictly guarantee <b>O(log N) worst-case time</b>.",
            body_style,
        )
    )
    story.append(Paragraph("Graph Algorithms & Pathfinding:", h2_style))
    story.append(
        Paragraph(
            "• <b>Breadth-First Search (BFS)</b>: Explores level by level using a FIFO queue. Time complexity is O(V + E). "
            "BFS is guaranteed to discover the shortest path in unweighted graphs.<br/>"
            "• <b>Depth-First Search (DFS)</b>: Explores as deep as possible along each branch before backtracking using recursion "
            "or an explicit stack. Used for topological sorting and cycle detection in directed graphs.<br/>"
            "• <b>Dijkstra's Algorithm</b>: Solves the single-source shortest path problem for graphs with non-negative edge weights. "
            "Using a Min-Heap priority queue, Dijkstra's algorithm achieves a time complexity of <b>O((V + E) log V)</b>.",
            body_style,
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Course Syllabus, Grading, & Policies
    # =========================================================================
    story.append(Paragraph("CS101: Course Syllabus & Administrative Policies", title_style))
    story.append(
        Paragraph(
            "<b>Course Instructor:</b> Professor Alan Turing | Office: Turing Hall 402<br/>"
            "<b>Teaching Assistant:</b> Grace Hopper | Office Hours: Wednesdays 3:00 PM - 5:00 PM<br/>"
            "<b>Lecture Schedule:</b> Mondays and Wednesdays 10:00 AM - 11:30 AM in Auditorium B",
            body_style,
        )
    )
    story.append(Paragraph("Grading Breakdown:", h2_style))

    grade_table_data = [
        ["Component", "Weight", "Description"],
        ["Programming Projects (4)", "40%", "Hands-on implementation of BST, Graph algorithms, and Caching"],
        ["Midterm Examination", "20%", "Held on Week 7 covering Units 1 and 2 (Linear structures & Big-O)"],
        ["Final Comprehensive Exam", "30%", "Scheduled during University Finals Week, covers all 4 units"],
        ["Weekly Quizzes & Labs", "10%", "Online multiple-choice quizzes and laboratory participation"],
    ]
    gt = Table(grade_table_data, colWidths=[150, 70, 280])
    gt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C5282")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#EDF2F7")),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
            ]
        )
    )
    story.append(gt)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Assignment Late Policy:", h2_style))
    story.append(
        Paragraph(
            "All programming projects are due at 11:59 PM on the specified due date. "
            "Late submissions are penalized at a rate of <b>10% per calendar day</b> for up to 3 days (72 hours). "
            "Submissions submitted more than 72 hours late will automatically receive a <b>grade of 0%</b>, "
            "unless an official university medical extension is granted in advance.",
            body_style,
        )
    )
    story.append(Paragraph("Academic Integrity Code:", h2_style))
    story.append(
        Paragraph(
            "All submitted code must be your own individual work. Copying code from classmates, online repositories, "
            "or unauthorized generative tools without attribution is considered a direct violation of the honor code. "
            "First offenses result in a zero on the assignment and a formal letter to the Dean of Academic Affairs.",
            body_style,
        )
    )

    doc.build(story)
    return str(dest.resolve())


if __name__ == "__main__":
    path = generate_sample_pdf()
    print(f"Sample PDF successfully generated at: {path}")
