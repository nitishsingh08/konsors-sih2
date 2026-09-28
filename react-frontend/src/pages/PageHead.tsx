/** the title and standfirst every full page starts with, with an optional mark */
export function PageHead({ title, lead, mark }: { title: string; lead: string; mark?: string }) {
  return (
    <div>
      <h1>
        {mark && (
          <img
            src={mark}
            alt=""
            width={32}
            height={32}
            style={{ verticalAlign: '-0.16em', marginRight: 10 }}
          />
        )}
        {title}
      </h1>
      <p className="lead">{lead}</p>
    </div>
  );
}
