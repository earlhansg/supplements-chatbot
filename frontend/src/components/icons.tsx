/**
 * Inline icons. Hand-rolled rather than pulling in an icon package for a dozen
 * glyphs; all share a 24x24 stroke grid so they line up at any size.
 */

type IconProps = React.SVGProps<SVGSVGElement>;

function Icon({ children, ...props }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}

export const DatabaseIcon = (props: IconProps) => (
  <Icon {...props}>
    <ellipse cx="12" cy="5.5" rx="7.5" ry="3" />
    <path d="M4.5 5.5v6c0 1.66 3.36 3 7.5 3s7.5-1.34 7.5-3v-6" />
    <path d="M4.5 11.5v7c0 1.66 3.36 3 7.5 3s7.5-1.34 7.5-3v-7" />
  </Icon>
);

export const ChatIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M20 12a7.5 7.5 0 0 1-10.9 6.7L4 20l1.3-4.1A7.5 7.5 0 1 1 20 12Z" />
  </Icon>
);

export const ActivityIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M3 12h4l2.5-7 4 14L16 12h5" />
  </Icon>
);

export const SendIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4.5 12h6" />
    <path d="m4 5 15 7-15 7 2-7Z" />
  </Icon>
);

export const RefreshIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M20 11a8 8 0 1 0-.6 4" />
    <path d="M20 4v6h-6" />
  </Icon>
);

export const TrashIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M4 7h16" />
    <path d="M9 7V5h6v2" />
    <path d="M6 7v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V7" />
    <path d="M10 11v6M14 11v6" />
  </Icon>
);

export const CloseIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M6 6l12 12M18 6 6 18" />
  </Icon>
);

export const BoltIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z" />
  </Icon>
);

export const AlertIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="M12 3.5 2.5 20h19L12 3.5Z" />
    <path d="M12 10v4.5M12 17.5h.01" />
  </Icon>
);

export const ChevronIcon = (props: IconProps) => (
  <Icon {...props}>
    <path d="m7 10 5 5 5-5" />
  </Icon>
);

export const SearchIcon = (props: IconProps) => (
  <Icon {...props}>
    <circle cx="11" cy="11" r="6.5" />
    <path d="m16 16 4.5 4.5" />
  </Icon>
);
